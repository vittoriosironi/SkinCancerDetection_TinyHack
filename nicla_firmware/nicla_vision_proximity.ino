/*
  Nicla Vision skin-cancer triage firmware with proximity sensor (hackathon prototype)

  NEW FEATURES:
  - Integrates VL53L1X proximity sensor to ensure correct camera distance (3-5 cm)
  - Yellow LED indicates distance is not optimal
  - Blue LED + processing only when distance is correct
  - Prevents captures at incorrect distances, improving diagnostic quality

  Responsibilities:
  - Continuously monitor distance using VL53L1X proximity sensor
  - Show yellow LED when distance is outside 3-5 cm range
  - When distance is optimal (3-5 cm):
    * Turn on blue LED
    * Capture and process image
    * Run inference with Zant model
    * Show red LED + upload if suspicious lesion detected
    * Show green LED if lesion is clear
  
  Hardware requirements:
  - Arduino Nicla Vision with VL53L1X proximity sensor (built-in)
  - WiFi connection for image upload
  - Zant-generated ML model library
*/

#include <Arduino.h>
#include <WiFiNINA.h>
#include <Arduino_H7_Vision.h>
#include <Nicla_System.h>
#include <VL53L1X.h>  // Proximity sensor library

#include <lib_zant.h>

// -----------------------------------------------------------------------------
// Wi-Fi & server configuration
// -----------------------------------------------------------------------------
constexpr char WIFI_SSID[] = "toolbox";
constexpr char WIFI_PASS[] = "Toolbox.Torino";

constexpr char SERVER_HOST[] = "110.100.15.27";
constexpr uint16_t SERVER_PORT = 8000;
constexpr char SERVER_ENDPOINT[] = "/ingest";

// -----------------------------------------------------------------------------
// Proximity sensor configuration
// -----------------------------------------------------------------------------
constexpr uint16_t MIN_DISTANCE_MM = 30;   // 3 cm minimum distance
constexpr uint16_t MAX_DISTANCE_MM = 50;   // 5 cm maximum distance
constexpr uint16_t DISTANCE_TIMEOUT_MS = 100;  // Sensor read timeout
constexpr unsigned long DISTANCE_CHECK_INTERVAL_MS = 200;  // Check every 200ms

// -----------------------------------------------------------------------------
// Timing & inference parameters
// -----------------------------------------------------------------------------
constexpr float ALERT_THRESHOLD = 0.65f;
constexpr unsigned long NEXT_CASE_DELAY_MS = 4000;
constexpr unsigned long WIFI_RETRY_DELAY_MS = 5000;

// Preview dimensions used for inference
constexpr int MODEL_IMAGE_WIDTH = 96;
constexpr int MODEL_IMAGE_HEIGHT = 96;
constexpr int MODEL_IMAGE_CHANNELS = 3;
constexpr int MODEL_INPUT_SIZE = MODEL_IMAGE_WIDTH * MODEL_IMAGE_HEIGHT * MODEL_IMAGE_CHANNELS;
static uint32_t MODEL_INPUT_SHAPE[] = {1, MODEL_IMAGE_CHANNELS, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH};
constexpr size_t MODEL_INPUT_DIMS = sizeof(MODEL_INPUT_SHAPE) / sizeof(MODEL_INPUT_SHAPE[0]);
constexpr int MODEL_OUTPUT_CLASSES = 2;
constexpr float PIXEL_SCALE = 1.0f / 255.0f;

// Camera configuration
constexpr CameraResolution CAMERA_INFERENCE_RES = CAMERA_QVGA;
constexpr CameraFormat CAMERA_INFERENCE_FMT = CAMERA_RGB565;
constexpr CameraResolution CAMERA_UPLOAD_RES = CAMERA_HD;
constexpr CameraFormat CAMERA_UPLOAD_FMT = CAMERA_JPEG;

constexpr int CAMERA_WIDTH_QVGA = 320;
constexpr int CAMERA_HEIGHT_QVGA = 240;

// Buffers
static float model_input_buffer[MODEL_INPUT_SIZE];
static float* model_output = nullptr;

// Proximity sensor instance
VL53L1X proximitySensor;

// State tracking
enum SystemState {
  STATE_WAITING_DISTANCE,    // Yellow LED - waiting for correct distance
  STATE_PROCESSING,          // Blue LED - capturing and processing
  STATE_ALERT,               // Red LED - suspicious lesion detected
  STATE_CLEAR                // Green LED - lesion is clear
};

SystemState currentState = STATE_WAITING_DISTANCE;
unsigned long lastDistanceCheck = 0;

// -----------------------------------------------------------------------------
// LED helpers (active low on Nicla Vision)
// -----------------------------------------------------------------------------
void setLedColor(bool red_on, bool green_on, bool blue_on) {
  digitalWrite(LEDR, red_on ? LOW : HIGH);
  digitalWrite(LEDG, green_on ? LOW : HIGH);
  digitalWrite(LEDB, blue_on ? LOW : HIGH);
}

void indicateIdle() { setLedColor(false, false, false); }
void indicateWaitingDistance() { 
  // Yellow LED = Red + Green
  digitalWrite(LEDR, LOW);
  digitalWrite(LEDG, LOW);
  digitalWrite(LEDB, HIGH);
}
void indicateProcessing() { setLedColor(false, false, true); }   // blue
void indicateClear() { setLedColor(false, true, false); }        // green
void indicateAlert() { setLedColor(true, false, false); }        // red

// -----------------------------------------------------------------------------
// Proximity sensor helpers
// -----------------------------------------------------------------------------
bool initProximitySensor() {
  Wire1.begin();
  Wire1.setClock(400000); // I2C fast mode
  
  proximitySensor.setBus(&Wire1);
  
  if (!proximitySensor.init()) {
    Serial.println("ERROR: Failed to initialize VL53L1X proximity sensor");
    return false;
  }
  
  proximitySensor.setDistanceMode(VL53L1X::Short);
  proximitySensor.setMeasurementTimingBudget(50000);  // 50ms timing budget
  proximitySensor.startContinuous(50);  // Read every 50ms
  
  Serial.println("Proximity sensor initialized successfully");
  return true;
}

uint16_t readDistance() {
  proximitySensor.read();
  
  if (proximitySensor.timeoutOccurred()) {
    Serial.println("WARNING: Proximity sensor timeout");
    return 0xFFFF;  // Return max value on timeout
  }
  
  return proximitySensor.ranging_data.range_mm;
}

bool isDistanceOptimal(uint16_t distance_mm) {
  if (distance_mm == 0xFFFF) {
    return false;  // Invalid reading
  }
  return (distance_mm >= MIN_DISTANCE_MM && distance_mm <= MAX_DISTANCE_MM);
}

// -----------------------------------------------------------------------------
// Wi-Fi connection
// -----------------------------------------------------------------------------
void connectToWiFi() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  Serial.print("Connecting to WiFi: ");
  Serial.println(WIFI_SSID);

  WiFi.disconnect();

  while (WiFi.begin(WIFI_SSID, WIFI_PASS) != WL_CONNECTED) {
    Serial.println("  • Connection failed, retrying...");
    delay(WIFI_RETRY_DELAY_MS);
  }

  Serial.print("Connected! IP address: ");
  Serial.println(WiFi.localIP());
}

// -----------------------------------------------------------------------------
// Image processing: RGB565 to model input
// -----------------------------------------------------------------------------
void downscaleToModelInput(const uint8_t* rgb565, int src_width, int src_height) {
  const int stride_x = (src_width << 8) / MODEL_IMAGE_WIDTH;
  const int stride_y = (src_height << 8) / MODEL_IMAGE_HEIGHT;

  for (int y = 0; y < MODEL_IMAGE_HEIGHT; ++y) {
    const int src_y = (y * stride_y) >> 8;
    for (int x = 0; x < MODEL_IMAGE_WIDTH; ++x) {
      const int src_x = (x * stride_x) >> 8;
      const uint16_t pixel = reinterpret_cast<const uint16_t*>(rgb565)[src_y * src_width + src_x];
      const uint8_t r5 = (pixel >> 11) & 0x1F;
      const uint8_t g6 = (pixel >> 5) & 0x3F;
      const uint8_t b5 = pixel & 0x1F;

      const uint8_t r8 = static_cast<uint8_t>((r5 << 3) | (r5 >> 2));
      const uint8_t g8 = static_cast<uint8_t>((g6 << 2) | (g6 >> 4));
      const uint8_t b8 = static_cast<uint8_t>((b5 << 3) | (b5 >> 2));

      const size_t base_index = static_cast<size_t>(y) * MODEL_IMAGE_WIDTH * MODEL_IMAGE_CHANNELS +
                static_cast<size_t>(x) * MODEL_IMAGE_CHANNELS;

      model_input_buffer[base_index + 0] = static_cast<float>(r8) * PIXEL_SCALE;
      model_input_buffer[base_index + 1] = static_cast<float>(g8) * PIXEL_SCALE;
      model_input_buffer[base_index + 2] = static_cast<float>(b8) * PIXEL_SCALE;
    }
  }
}

// -----------------------------------------------------------------------------
// ML Inference
// -----------------------------------------------------------------------------
float runInference() {
  const int status = predict(model_input_buffer, MODEL_INPUT_SHAPE,
                             static_cast<uint32_t>(MODEL_INPUT_DIMS), &model_output);
  if (status != 0 || model_output == nullptr) {
    Serial.print("ERROR: predict() failed with code ");
    Serial.println(status);
    return 0.0f;
  }

  const float suspicious_score = (MODEL_OUTPUT_CLASSES > 1) ? model_output[1] : model_output[0];
  return constrain(suspicious_score, 0.0f, 1.0f);
}

// -----------------------------------------------------------------------------
// Server upload
// -----------------------------------------------------------------------------
bool sendAlertToServer(const uint8_t* jpeg_data, size_t jpeg_len, float score) {
  WiFiClient client;
  if (!client.connect(SERVER_HOST, SERVER_PORT)) {
    Serial.println("ERROR: cannot reach server");
    return false;
  }

  char metadata[256];
  snprintf(metadata, sizeof(metadata),
           "{\"device_id\":\"nicla-vision\",\"score\":%.3f,\"timestamp\":%lu}",
           score, millis());

  const char boundary[] = "----NICLAHACKATHONBOUNDARY";

  String body_header;
  body_header.reserve(256);
  body_header += "--";
  body_header += boundary;
  body_header += "\r\nContent-Disposition: form-data; name=\"metadata\"\r\n\r\n";
  body_header += metadata;
  body_header += "\r\n--";
  body_header += boundary;
  body_header += "\r\nContent-Disposition: form-data; name=\"image\"; filename=\"capture.jpg\"\r\n";
  body_header += "Content-Type: image/jpeg\r\n\r\n";

  const String body_footer = String("\r\n--") + boundary + "--\r\n";

  const size_t content_length = body_header.length() + jpeg_len + body_footer.length();

  client.print("POST ");
  client.print(SERVER_ENDPOINT);
  client.println(" HTTP/1.1");
  client.print("Host: ");
  client.print(SERVER_HOST);
  client.print(":");
  client.println(SERVER_PORT);
  client.println("Connection: close");
  client.print("Content-Type: multipart/form-data; boundary=");
  client.println(boundary);
  client.print("Content-Length: ");
  client.println(content_length);
  client.println();

  client.print(body_header);
  client.write(jpeg_data, jpeg_len);
  client.print(body_footer);

  client.flush();

  unsigned long deadline = millis() + 5000;
  while (client.connected() && millis() < deadline) {
    while (client.available()) {
      Serial.write(client.read());
    }
  }
  client.stop();
  return true;
}

// -----------------------------------------------------------------------------
// Camera capture helpers
// -----------------------------------------------------------------------------
bool captureInferenceFrame() {
  CamImage img = Camera.grab(CAMERA_INFERENCE_RES, CAMERA_INFERENCE_FMT);
  if (!img.isAvailable()) {
    Serial.println("ERROR: failed to capture inference frame");
    return false;
  }
  downscaleToModelInput(img.buffer(), img.width(), img.height());
  return true;
}

struct CapturePayload {
  const uint8_t* data;
  size_t length;
};

bool captureHighResolutionJPEG(CapturePayload& out) {
  CamImage img = Camera.grab(CAMERA_UPLOAD_RES, CAMERA_UPLOAD_FMT);
  if (!img.isAvailable()) {
    Serial.println("ERROR: high-resolution capture failed");
    return false;
  }
  out.data = img.buffer();
  out.length = img.size();
  return true;
}

// -----------------------------------------------------------------------------
// Main processing workflow (triggered when distance is optimal)
// -----------------------------------------------------------------------------
void processCapture() {
  indicateProcessing();
  
  Serial.println("Distance optimal - Processing capture...");

  if (!captureInferenceFrame()) {
    indicateIdle();
    delay(500);
    currentState = STATE_WAITING_DISTANCE;
    return;
  }

  const float suspicious_score = runInference();
  Serial.print("Inference score: ");
  Serial.println(suspicious_score, 3);

  if (suspicious_score >= ALERT_THRESHOLD) {
    currentState = STATE_ALERT;
    indicateAlert();
    
    CapturePayload payload{};
    if (captureHighResolutionJPEG(payload)) {
      Serial.println("Uploading suspicious capture...");
      if (sendAlertToServer(payload.data, payload.length, suspicious_score)) {
        Serial.println("Upload complete");
      } else {
        Serial.println("Upload failed");
      }
    }
    
    delay(NEXT_CASE_DELAY_MS);
    currentState = STATE_WAITING_DISTANCE;
    indicateIdle();
  } else {
    currentState = STATE_CLEAR;
    indicateClear();
    
    delay(NEXT_CASE_DELAY_MS);
    currentState = STATE_WAITING_DISTANCE;
    indicateIdle();
  }
}

// -----------------------------------------------------------------------------
// Setup
// -----------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ;
  }

  Serial.println("=== Nicla Vision Proximity-Based Skin Cancer Triage ===");

  nicla::begin();
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  indicateIdle();

  // Initialize proximity sensor
  if (!initProximitySensor()) {
    Serial.println("FATAL: Proximity sensor initialization failed");
    while (true) {
      // Blink red LED to indicate error
      setLedColor(true, false, false);
      delay(500);
      setLedColor(false, false, false);
      delay(500);
    }
  }

  // Initialize camera
  if (!Camera.begin()) {
    Serial.println("FATAL: Camera did not start");
    while (true) {
      delay(1000);
    }
  }
  Camera.setAutoExposure(true);
  Camera.setAutoWhiteBalance(true);

  // Connect to WiFi
  connectToWiFi();

  Serial.println("System ready. Waiting for optimal distance (3-5 cm)...");
  indicateWaitingDistance();
}

// -----------------------------------------------------------------------------
// Main loop
// -----------------------------------------------------------------------------
void loop() {
  // Ensure WiFi is connected
  connectToWiFi();

  // Check distance at regular intervals
  unsigned long currentTime = millis();
  
  if (currentTime - lastDistanceCheck >= DISTANCE_CHECK_INTERVAL_MS) {
    lastDistanceCheck = currentTime;
    
    uint16_t distance = readDistance();
    
    // Debug output
    if (distance != 0xFFFF) {
      Serial.print("Distance: ");
      Serial.print(distance);
      Serial.println(" mm");
    }
    
    // State machine based on distance and current state
    if (currentState == STATE_WAITING_DISTANCE) {
      if (isDistanceOptimal(distance)) {
        // Distance is good - start processing
        currentState = STATE_PROCESSING;
        processCapture();
      } else {
        // Distance not optimal - keep showing yellow LED
        indicateWaitingDistance();
      }
    }
    // If we're in ALERT or CLEAR states, the processCapture() function
    // will handle the transition back to WAITING_DISTANCE
  }
  
  // Small delay to prevent tight loop
  delay(10);
}
