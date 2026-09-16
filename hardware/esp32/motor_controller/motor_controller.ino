/*
 * Vision-Guided Person-Following Rover
 * ESP32 Motor Controller
 *
 * ESP32 -> L298N -> Two DC Geared Motors
 *
 * Serial command format:
 *     <linear_velocity> <angular_velocity>
 *
 * Examples:
 *     0.20 0.00
 *     0.00 0.50
 *     0.00 0.00
 *
 * The Raspberry Pi / ROS 2 side will eventually send these commands
 * to the ESP32.
 */

#include <Arduino.h>

// --------------------------------------------------
// L298N pin configuration
// --------------------------------------------------

const int ENA = 25;
const int IN1 = 26;
const int IN2 = 27;

const int IN3 = 14;
const int IN4 = 16;
const int ENB = 17;

// --------------------------------------------------
// Robot parameters
// --------------------------------------------------

const float WHEEL_SEPARATION = 0.32f;  // metres
const float MAX_WHEEL_SPEED = 0.50f;   // m/s

// PWM configuration
const int PWM_FREQUENCY = 20000;
const int PWM_RESOLUTION = 8;

// ESP32 Arduino Core 3.x uses pin-based LEDC PWM.

// --------------------------------------------------
// Safety
// --------------------------------------------------

const unsigned long COMMAND_TIMEOUT_MS = 500;

// --------------------------------------------------
// State
// --------------------------------------------------

unsigned long lastCommandTime = 0;

// --------------------------------------------------
// Set motor A
// --------------------------------------------------

void setMotorA(float speed)
{
    speed = constrain(speed, -1.0f, 1.0f);

    if (speed > 0.0f)
    {
        digitalWrite(IN1, HIGH);
        digitalWrite(IN2, LOW);
    }
    else if (speed < 0.0f)
    {
        digitalWrite(IN1, LOW);
        digitalWrite(IN2, HIGH);
    }
    else
    {
        digitalWrite(IN1, LOW);
        digitalWrite(IN2, LOW);
    }

    int pwm = (int)(fabs(speed) * 255.0f);
    ledcWrite(ENA, pwm);
}

// --------------------------------------------------
// Set motor B
// --------------------------------------------------

void setMotorB(float speed)
{
    speed = constrain(speed, -1.0f, 1.0f);

    if (speed > 0.0f)
    {
        digitalWrite(IN3, HIGH);
        digitalWrite(IN4, LOW);
    }
    else if (speed < 0.0f)
    {
        digitalWrite(IN3, LOW);
        digitalWrite(IN4, HIGH);
    }
    else
    {
        digitalWrite(IN3, LOW);
        digitalWrite(IN4, LOW);
    }

    int pwm = (int)(fabs(speed) * 255.0f);
    ledcWrite(ENB, pwm);
}

// --------------------------------------------------
// Stop both motors
// --------------------------------------------------

void stopMotors()
{
    setMotorA(0.0f);
    setMotorB(0.0f);
}

// --------------------------------------------------
// Convert robot velocity to wheel velocities
// --------------------------------------------------

void setVelocity(float linearVelocity, float angularVelocity)
{
    /*
     * Differential-drive equations:
     *
     * v_left  = v - (w * L / 2)
     * v_right = v + (w * L / 2)
     */

    float leftVelocity =
        linearVelocity -
        (angularVelocity * WHEEL_SEPARATION / 2.0f);

    float rightVelocity =
        linearVelocity +
        (angularVelocity * WHEEL_SEPARATION / 2.0f);

    // Normalize if either wheel exceeds the allowed range
    float maximum =
        max(fabs(leftVelocity), fabs(rightVelocity));

    if (maximum > MAX_WHEEL_SPEED)
    {
        leftVelocity /= maximum;
        rightVelocity /= maximum;

        leftVelocity *= MAX_WHEEL_SPEED;
        rightVelocity *= MAX_WHEEL_SPEED;
    }

    // Convert wheel velocity to normalized motor speed
    float leftMotor =
        leftVelocity / MAX_WHEEL_SPEED;

    float rightMotor =
        rightVelocity / MAX_WHEEL_SPEED;

    setMotorA(leftMotor);
    setMotorB(rightMotor);
}

// --------------------------------------------------
// Process serial command
// --------------------------------------------------

void processCommand(String command)
{
    command.trim();

    if (command.length() == 0)
    {
        return;
    }

    float linearVelocity;
    float angularVelocity;

    int values =
        sscanf(
            command.c_str(),
            "%f %f",
            &linearVelocity,
            &angularVelocity
        );

    if (values == 2)
    {
        setVelocity(linearVelocity, angularVelocity);

        lastCommandTime = millis();

        Serial.print("CMD: ");
        Serial.print(linearVelocity, 3);
        Serial.print(" ");
        Serial.println(angularVelocity, 3);
    }
    else
    {
        Serial.println("ERROR: Expected <linear> <angular>");
    }
}

// --------------------------------------------------
// Setup
// --------------------------------------------------

void setup()
{
    Serial.begin(115200);

    pinMode(IN1, OUTPUT);
    pinMode(IN2, OUTPUT);
    pinMode(IN3, OUTPUT);
    pinMode(IN4, OUTPUT);

    // Configure PWM
    ledcAttach(ENA, PWM_FREQUENCY, PWM_RESOLUTION);
    ledcAttach(ENB, PWM_FREQUENCY, PWM_RESOLUTION);

    stopMotors();

    lastCommandTime = millis();

    Serial.println();
    Serial.println("================================");
    Serial.println("ESP32 Motor Controller");
    Serial.println("================================");
    Serial.println("Ready");
}

// --------------------------------------------------
// Main loop
// --------------------------------------------------

void loop()
{
    static String command = "";

    while (Serial.available())
    {
        char character = Serial.read();

        if (character == '\n')
        {
            processCommand(command);
            command = "";
        }
        else if (character != '\r')
        {
            command += character;
        }
    }

    // Safety timeout
    if (millis() - lastCommandTime > COMMAND_TIMEOUT_MS)
    {
        stopMotors();
    }

    delay(5);
}
