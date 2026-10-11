#include <Arduino.h>

// L298N pins
const int ENA = 25;
const int IN1 = 26;
const int IN2 = 27;

const int IN3 = 14;
const int IN4 = 16;
const int ENB = 17;

void setup()
{
    Serial.begin(115200);

    pinMode(ENA, OUTPUT);
    pinMode(IN1, OUTPUT);
    pinMode(IN2, OUTPUT);

    pinMode(ENB, OUTPUT);
    pinMode(IN3, OUTPUT);
    pinMode(IN4, OUTPUT);

    // Stop everything initially
    digitalWrite(ENA, LOW);
    digitalWrite(ENB, LOW);

    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, LOW);

    delay(2000);

    Serial.println("Starting forward motor test...");

    // LEFT MOTOR FORWARD
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
    digitalWrite(ENA, HIGH);

    // RIGHT MOTOR FORWARD
    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);
    digitalWrite(ENB, HIGH);

    Serial.println("MOTORS FORWARD");
}

void loop()
{
    // Keep both motors running
    digitalWrite(ENA, HIGH);
    digitalWrite(ENB, HIGH);

    delay(100);
}