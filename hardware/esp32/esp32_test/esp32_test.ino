void setup() {
  Serial.begin(115200);
  Serial.println("ESP32 hardware test OK");
}

void loop() {
  delay(1000);
  Serial.println("ESP32 running");
}
