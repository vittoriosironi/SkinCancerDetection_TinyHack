/*
  Nicla Vision skin-cancer triage firmware (hackathon prototype)

  Responsibilities:
  - Wake the Himax camera, light up the blue LED while a frame is captured.
  - Downscale the frame to 96x96 RGB (quantized) and feed it to a Zant-generated model.
  - If the model flags a suspicious lesion, turn on the red LED and upload the full-resolution image
    together with JSON metadata to the ingestion server (Flask app built earlier).
  - Otherwise, turn on the green LED to let the clinician know they can move to the next mole.

  Assumptions:
  - A Zant-generated static library (`lib_zant`) has been flashed alongside this sketch.
  - The Nicla Vision is programmed with the Arduino IDE or Arduino CLI using the "Nicla Vision" core.
  - Networking uses the onboard NINA-W102 (WiFiNINA library).
  - Camera API provided by Arduino_H7_Vision (bundled with the Nicla Vision core >=1.0.4).
  - LEDs are active-low (write LOW to switch on, HIGH to switch off).

  NOTE: Camera helper calls may require minor adjustments depending on your board package version.
        This sketch focuses on plumbing the end-to-end logic for the hackathon.
*/

#include <Arduino.h>
#include <WiFiNINA.h>
#include <Arduino_H7_Vision.h>
#include <Nicla_System.h>

#include <lib_zant.h>

// -----------------------------------------------------------------------------
// Wi-Fi & server configuration (edit for your network / backend)
// -----------------------------------------------------------------------------
constexpr char WIFI_SSID[] = "toolbox";
constexpr char WIFI_PASS[] = "Toolbox.Torino";

constexpr char SERVER_HOST[] = "110.100.15.27";  // Flask ingestion server IP
constexpr uint16_t SERVER_PORT = 8000;
constexpr char SERVER_ENDPOINT[] = "/ingest";

// -----------------------------------------------------------------------------
// Timing & inference parameters
// -----------------------------------------------------------------------------
constexpr float ALERT_THRESHOLD = 0.65f;          // score above which we treat as suspicious
constexpr unsigned long NEXT_CASE_DELAY_MS = 4000;  // downtime before next scan (green LED)
constexpr unsigned long WIFI_RETRY_DELAY_MS = 5000;

// Preview dimensions used for inference (must match the model's input shape)
constexpr int MODEL_IMAGE_WIDTH = 96;
constexpr int MODEL_IMAGE_HEIGHT = 96;
constexpr int MODEL_IMAGE_CHANNELS = 3;
constexpr int MODEL_INPUT_SIZE = MODEL_IMAGE_WIDTH * MODEL_IMAGE_HEIGHT * MODEL_IMAGE_CHANNELS;
static uint32_t MODEL_INPUT_SHAPE[] = {1, MODEL_IMAGE_CHANNELS, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH};
constexpr size_t MODEL_INPUT_DIMS = sizeof(MODEL_INPUT_SHAPE) / sizeof(MODEL_INPUT_SHAPE[0]);
constexpr int MODEL_OUTPUT_CLASSES = 2;  // adjust if your Zant model has a different number of outputs
constexpr float PIXEL_SCALE = 1.0f / 255.0f;

// Camera configuration (Nicla Vision camera supports up to 2MP JPEG)
constexpr CameraResolution CAMERA_INFERENCE_RES = CAMERA_QVGA;     // 320x240 RGB565
constexpr CameraFormat CAMERA_INFERENCE_FMT = CAMERA_RGB565;       // used for downscaling
constexpr CameraResolution CAMERA_UPLOAD_RES = CAMERA_HD;          // 1920x1080 JPEG (approx 2MP)
constexpr CameraFormat CAMERA_UPLOAD_FMT = CAMERA_JPEG;

constexpr int CAMERA_WIDTH_QVGA = 320;
constexpr int CAMERA_HEIGHT_QVGA = 240;

// Buffers
static float model_input_buffer[MODEL_INPUT_SIZE];
static float* model_output = nullptr;

// -----------------------------------------------------------------------------
// LED helpers (active low on Nicla Vision)
// -----------------------------------------------------------------------------
void setLedColor(bool red_on, bool green_on, bool blue_on) {
  digitalWrite(LEDR, red_on ? LOW : HIGH);
  digitalWrite(LEDG, green_on ? LOW : HIGH);
  digitalWrite(LEDB, blue_on ? LOW : HIGH);
}

void indicateIdle() { setLedColor(false, false, false); }
void indicateCapturing() { setLedColor(false, false, true); }   // blue
void indicateClear() { setLedColor(false, true, false); }       // green
void indicateAlert() { setLedColor(true, false, false); }       // red

// -----------------------------------------------------------------------------
// Utility: connect to Wi-Fi network
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
// Utility: convert RGB565 inference frame to 96x96 RGB float input
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
// Utility: run inference and return suspicious score (0-1)
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
// Utility: send high-resolution JPEG frame to the ingestion server
// -----------------------------------------------------------------------------
bool sendAlertToServer(const uint8_t* jpeg_data, size_t jpeg_len, float score) {
  WiFiClient client;
  if (!client.connect(SERVER_HOST, SERVER_PORT)) {
    Serial.println("ERROR: cannot reach server");
    return false;
  }

  // Compose metadata JSON (keep it tiny to save RAM)
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

  // HTTP request headers
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

  // Body parts
  client.print(body_header);
  client.write(jpeg_data, jpeg_len);
  client.print(body_footer);

  client.flush();

  // Optional: read server response (for debugging)
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
// Capture helper wrappers (grabs inference + upload frames)
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
// Setup & main loop
// -----------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ;
  }

  nicla::begin();
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  indicateIdle();

  if (!Camera.begin()) {
    Serial.println("FATAL: Camera did not start");
    while (true) {
      delay(1000);
    }
  }
  Camera.setAutoExposure(true);
  Camera.setAutoWhiteBalance(true);

  connectToWiFi();
}

void loop() {
  connectToWiFi();

  indicateCapturing();

  if (!captureInferenceFrame()) {
    indicateIdle();
    delay(500);
    return;
  }

  const float suspicious_score = runInference();
  Serial.print("Inference score: ");
  Serial.println(suspicious_score, 3);

  if (suspicious_score >= ALERT_THRESHOLD) {
    indicateAlert();
    CapturePayload payload{};
    if (captureHighResolutionJPEG(payload)) {
      Serial.println("Uploading suspicious capture...\n");
      if (sendAlertToServer(payload.data, payload.length, suspicious_score)) {
        Serial.println("Upload complete");
      } else {
        Serial.println("Upload failed");
      }
    }
    delay(NEXT_CASE_DELAY_MS);
    indicateIdle();
  } else {
    indicateClear();
    delay(NEXT_CASE_DELAY_MS);
    indicateIdle();
  }
}
