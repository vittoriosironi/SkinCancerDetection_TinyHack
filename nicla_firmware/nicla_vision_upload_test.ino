/*
  Nicla Vision quick upload test (hackathon helper)

  Responsibilities:
  - Power the Himax camera and grab a high-resolution JPEG frame.
  - Blink the blue LED while capturing.
  - Push the JPEG plus a small JSON metadata blob to the Flask ingestion server.
  - Toggle the green LED on success and the red LED on failure.

  Assumptions:
  - Same Wi-Fi credentials and ingestion endpoint as the main firmware.
  - The ingestion server accepts multipart/form-data with fields "metadata" and "image".
  - Designed purely for plumbing validation: no inference, no Zant dependencies.
*/

#include <Arduino.h>
#include <WiFiNINA.h>
#include <Arduino_H7_Vision.h>
#include <Nicla_System.h>

// -----------------------------------------------------------------------------
// Wi-Fi & server configuration (edit for your network / backend)
// -----------------------------------------------------------------------------
constexpr char WIFI_SSID[] = "toolbox";
constexpr char WIFI_PASS[] = "Toolbox.Torino";

constexpr char SERVER_HOST[] = "110.100.15.27";  // Flask ingestion server IP
constexpr uint16_t SERVER_PORT = 8000;
constexpr char SERVER_ENDPOINT[] = "/ingest";

// -----------------------------------------------------------------------------
// Camera + timing configuration
// -----------------------------------------------------------------------------
constexpr CameraResolution CAMERA_UPLOAD_RES = CAMERA_HD;   // 1920x1080 JPEG
constexpr CameraFormat CAMERA_UPLOAD_FMT = CAMERA_JPEG;
constexpr unsigned long UPLOAD_INTERVAL_MS = 6000;         // wait before next capture
constexpr unsigned long WIFI_RETRY_DELAY_MS = 5000;

struct CapturePayload {
  const uint8_t* data;
  size_t length;
};

// -----------------------------------------------------------------------------
// LED helpers (Nicla Vision LEDs are active-low)
// -----------------------------------------------------------------------------
void setLedColor(bool red_on, bool green_on, bool blue_on) {
  digitalWrite(LEDR, red_on ? LOW : HIGH);
  digitalWrite(LEDG, green_on ? LOW : HIGH);
  digitalWrite(LEDB, blue_on ? LOW : HIGH);
}

void indicateIdle() { setLedColor(false, false, false); }
void indicateCapturing() { setLedColor(false, false, true); }
void indicateSuccess() { setLedColor(false, true, false); }
void indicateFailure() { setLedColor(true, false, false); }

// -----------------------------------------------------------------------------
// Wi-Fi connectivity helper
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
// Capture a high-resolution JPEG frame
// -----------------------------------------------------------------------------
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
// Upload helper (multipart/form-data)
// -----------------------------------------------------------------------------
bool uploadFrame(const CapturePayload& payload) {
  WiFiClient client;
  if (!client.connect(SERVER_HOST, SERVER_PORT)) {
    Serial.println("ERROR: cannot reach server");
    return false;
  }

  char metadata[256];
  snprintf(metadata, sizeof(metadata),
           "{\"device_id\":\"nicla-upload-test\",\"timestamp\":%lu}", millis());

  const char boundary[] = "----NICLAUPLOADTESTBOUNDARY";

  String body_header;
  body_header.reserve(256);
  body_header += "--";
  body_header += boundary;
  body_header += "\r\nContent-Disposition: form-data; name=\"metadata\"\r\n\r\n";
  body_header += metadata;
  body_header += "\r\n--";
  body_header += boundary;
  body_header +=
      "\r\nContent-Disposition: form-data; name=\"image\"; filename=\"capture.jpg\"\r\n";
  body_header += "Content-Type: image/jpeg\r\n\r\n";

  const String body_footer = String("\r\n--") + boundary + "--\r\n";
  const size_t content_length = body_header.length() + payload.length + body_footer.length();

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
  client.write(payload.data, payload.length);
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
// Setup & loop
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
  CapturePayload payload{};
  if (!captureHighResolutionJPEG(payload)) {
    indicateFailure();
    delay(UPLOAD_INTERVAL_MS);
    indicateIdle();
    return;
  }

  Serial.println("Captured frame, uploading...");
  if (uploadFrame(payload)) {
    Serial.println("Upload complete");
    indicateSuccess();
  } else {
    Serial.println("Upload failed");
    indicateFailure();
  }

  delay(UPLOAD_INTERVAL_MS);
  indicateIdle();
}
