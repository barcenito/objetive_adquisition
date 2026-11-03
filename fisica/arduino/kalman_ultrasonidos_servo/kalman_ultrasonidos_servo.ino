#include <Arduino.h>
#include <Servo.h>

// ============================================================================
// CONFIGURACION DE HARDWARE Y PINES
// ============================================================================
const int trigPin = 2;   // Pin Trigger del HC-SR04
const int echoPin = 3;   // Pin Echo del HC-SR04
const int servoPin = 4;  // Pin de control del servo (cable naranja)

Servo miServo;

// Limites mecanicos seguros para el servomotor
const int SERVO_MIN_ANGLE = 20;
const int SERVO_MAX_ANGLE = 160;

// Rango de deteccion valido del objetivo (en cm)
const float MIN_DISTANCE = 5.0;
const float MAX_DISTANCE = 45.0;

// Parametros de barrido de sector (Opcion A - Sector Centroid Tracking)
const float BASE_SECTOR_HALF_AMP = 10.0; // Amplitud base (+/- 10 grados alrededor del blanco)
const float MAX_SECTOR_HALF_AMP = 28.0;  // Amplitud maxima adaptativa para blancos rapidos
const float SWEEP_STEP = 1.8;            // Avance angular por paso (movimiento continuo suave)
const int STEP_DELAY_MS = 22;            // Cadencia de muestreo (~45 Hz, reposo acustico)

// ============================================================================
// FILTRO DE KALMAN 1D/2D PARA DISTANCIA Y ANGULO
// ============================================================================
// Modelo cinemático: x = [posicion, velocidad]
// - Distancia: r (cm), r_dot (cm/s)
// - Angulo: theta (deg), theta_dot (deg/s)
// ============================================================================
struct KalmanTracker1D {
  float pos;       // Estimacion de posicion
  float vel;       // Estimacion de velocidad (delta posicion / delta t)
  float p00, p01;  // Matriz de covarianza P
  float p10, p11;
  float q_acc;     // Ruido de proceso (aceleracion)
  float r_var;     // Varianza del ruido de medicion del sensor

  void init(float initial_pos, float q, float r) {
    pos = initial_pos;
    vel = 0.0;
    p00 = 4.0;  p01 = 0.0;
    p10 = 0.0;  p11 = 10.0;
    q_acc = q;
    r_var = r;
  }

  void predict(float dt) {
    // Proyeccion del estado: x = F * x
    pos += vel * dt;

    // Proyeccion de covarianza: P = F * P * F^T + Q
    float dt2 = dt * dt;
    float dt3 = dt2 * dt;
    float dt4 = dt3 * dt;

    float np00 = p00 + dt * (p10 + p01) + dt2 * p11 + q_acc * dt4 / 4.0;
    float np01 = p01 + dt * p11 + q_acc * dt3 / 2.0;
    float np10 = p10 + dt * p11 + q_acc * dt3 / 2.0;
    float np11 = p11 + q_acc * dt2;

    p00 = np00;
    p01 = np01;
    p10 = np10;
    p11 = np11;
  }

  void update(float z) {
    // Innovacion
    float y = z - pos;
    float S = p00 + r_var;
    float invS = 1.0 / S;

    float k0 = p00 * invS;
    float k1 = p10 * invS;

    pos += k0 * y;
    vel += k1 * y;

    float op00 = p00;
    float op01 = p01;
    float op10 = p10;
    float op11 = p11;

    p00 = (1.0 - k0) * op00;
    p01 = (1.0 - k0) * op01;
    p10 = op10 - k1 * op00;
    p11 = op11 - k1 * op01;
  }

  void dampVelocity(float factor) {
    vel *= factor;
  }
};

KalmanTracker1D kf_dist;
KalmanTracker1D kf_angle;

// ============================================================================
// MAQUINA DE ESTADOS Y CONTROL DEL SERVO (SECTOR CENTROID TRACKING)
// ============================================================================
enum TrackerMode {
  MODE_SEARCH,       // Busqueda panoramica de 20 a 160 deg
  MODE_TRACK_SECTOR  // Barrido de sector continuo y estimacion de centroide
};

TrackerMode currentMode = MODE_SEARCH;

float currentServoAngle = 90.0;
float searchDir = 1.0;
float sweepDir = 1.0;
float currentSectorAmp = BASE_SECTOR_HALF_AMP;

// Variables para deteccion de flancos acusticos y centroide
bool echoActive = false;
float echoStartAngle = 0.0;
float echoEndAngle = 0.0;
int lostSweepCount = 0;
const int MAX_LOST_SWEEPS = 6; // ~1.5s sin eco -> vuelve a busqueda panoramica

unsigned long lastLoopMs = 0;
unsigned long lastTelemetryMs = 0;

// ============================================================================
// FUNCIONES AUXILIARES DE MEDICION Y HARDWARE
// ============================================================================

// Lectura ultrasonica robusta con timeout seguro de 25000 µs (~4.2 metros max)
// Garantiza compatibilidad con todos los modulos HC-SR04 y clones sin perder pulsos
float readUltrasonic() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(4);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  long duration = pulseIn(echoPin, HIGH, 25000);
  if (duration <= 0) return -1.0;

  float d = duration * 0.034 / 2.0;
  return d;
}

// Control directo de posicion del servomotor con limites mecanicos
void setServoAngle(float angle) {
  angle = constrain(angle, (float)SERVO_MIN_ANGLE, (float)SERVO_MAX_ANGLE);
  currentServoAngle = angle;
  miServo.write((int)round(currentServoAngle));
}

// ============================================================================
// SETUP
// ============================================================================
void setup() {
  // Baudrate fijo a 9600 (sincronizado con monitor_speed en platformio.ini)
  Serial.begin(9600);

  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  miServo.attach(servoPin);
  setServoAngle(90.0);
  delay(400);

  // Inicializacion del Filtro de Kalman:
  // Distancia: q = 15.0, r = 0.25 (0.5 cm desv. estandar)
  // Angulo: q = 150.0 (alta agilidad ante cambios de direccion), r = 2.0 (ruido centroide)
  kf_dist.init(25.0, 15.0, 0.25);
  kf_angle.init(90.0, 150.0, 2.0);

  lastLoopMs = millis();
  lastTelemetryMs = millis();

  Serial.println(F("--- Tracker Sectorial Centroid Listo (9600 baud) ---"));
}

// ============================================================================
// LOOP PRINCIPAL
// ============================================================================
void loop() {
  unsigned long now = millis();
  float dt = (now - lastLoopMs) / 1000.0;
  if (dt < 0.001) dt = 0.001;
  lastLoopMs = now;

  // --------------------------------------------------------------------------
  // MODO 1: BUSQUEDA PANORAMICA (SEARCH)
  // --------------------------------------------------------------------------
  if (currentMode == MODE_SEARCH) {
    currentServoAngle += searchDir * 3.0;
    if (currentServoAngle >= SERVO_MAX_ANGLE) {
      currentServoAngle = SERVO_MAX_ANGLE;
      searchDir = -1.0;
    } else if (currentServoAngle <= SERVO_MIN_ANGLE) {
      currentServoAngle = SERVO_MIN_ANGLE;
      searchDir = 1.0;
    }

    setServoAngle(currentServoAngle);
    delay(35); // Pausa para movimiento del servo y reposo acustico

    float dist = readUltrasonic();

    // Telemetria periodica en busqueda para verificar mediciones en vivo
    if (now - lastTelemetryMs >= 200) {
      lastTelemetryMs = now;
      Serial.print(F("[BUSQUEDA] Ang: "));
      Serial.print((int)round(currentServoAngle));
      Serial.print(F(" deg | Eco: "));
      if (dist > 0) {
        Serial.print(dist, 1);
        Serial.print(F(" cm"));
        if (dist < MIN_DISTANCE || dist > MAX_DISTANCE) {
          Serial.print(F(" (fuera de rango [5-45cm])"));
        }
      } else {
        Serial.print(F("---"));
      }
      Serial.println();
    }

    if (dist >= MIN_DISTANCE && dist <= MAX_DISTANCE) {
      // Objetivo detectado: Fijar e iniciar seguimiento sectorial centrado
      kf_dist.init(dist, 15.0, 0.25);
      kf_angle.init(currentServoAngle, 150.0, 2.0);
      currentSectorAmp = BASE_SECTOR_HALF_AMP;
      sweepDir = 1.0;
      echoActive = false;
      lostSweepCount = 0;
      currentMode = MODE_TRACK_SECTOR;

      Serial.print(F(">>> Lock en Ang: "));
      Serial.print((int)round(currentServoAngle));
      Serial.print(F(" deg | Dist: "));
      Serial.print(dist, 1);
      Serial.println(F(" cm"));
    }

  // --------------------------------------------------------------------------
  // MODO 2: SEGUIMIENTO POR BARRIDO DE SECTOR Y CENTROIDE (TRACK_SECTOR)
  // --------------------------------------------------------------------------
  } else {
    // 1. Prediccion temporal del Filtro de Kalman
    kf_dist.predict(dt);
    kf_angle.predict(dt);
    kf_angle.pos = constrain(kf_angle.pos, (float)SERVO_MIN_ANGLE, (float)SERVO_MAX_ANGLE);

    // 2. Limites del sector de barrido centrados en la posicion predicha
    float leftLimit = max((float)SERVO_MIN_ANGLE, kf_angle.pos - currentSectorAmp);
    float rightLimit = min((float)SERVO_MAX_ANGLE, kf_angle.pos + currentSectorAmp);

    // 3. Avance suave y continuo del servomotor
    currentServoAngle += sweepDir * SWEEP_STEP;
    bool reachedTurnaround = false;

    if (sweepDir > 0 && currentServoAngle >= rightLimit) {
      currentServoAngle = rightLimit;
      sweepDir = -1.0;
      reachedTurnaround = true;
    } else if (sweepDir < 0 && currentServoAngle <= leftLimit) {
      currentServoAngle = leftLimit;
      sweepDir = 1.0;
      reachedTurnaround = true;
    }

    setServoAngle(currentServoAngle);
    delay(STEP_DELAY_MS);

    // 4. Muestreo ultrasonico en el angulo actual
    float dist = readUltrasonic();

    if (dist >= MIN_DISTANCE && dist <= MAX_DISTANCE) {
      lostSweepCount = 0;
      currentSectorAmp = BASE_SECTOR_HALF_AMP;
      kf_dist.update(dist);

      if (!echoActive) {
        // Flanco de subida: el haz acustico empieza a tocar el blanco
        echoActive = true;
        echoStartAngle = currentServoAngle;
      }

      // Si alcanzamos el extremo del barrido viendo aun el blanco,
      // el objetivo se desplaza activamente hacia ese lado (empuje inmediato)
      if (reachedTurnaround) {
        echoEndAngle = currentServoAngle;
        float centroid = (echoStartAngle + echoEndAngle) / 2.0;
        kf_angle.update(centroid);
        // sweepDir ya fue invertido, por lo que -sweepDir apunta a la direccion de escape
        kf_angle.pos = constrain(kf_angle.pos - sweepDir * 3.5, (float)SERVO_MIN_ANGLE, (float)SERVO_MAX_ANGLE);
        echoStartAngle = currentServoAngle; // Mantiene el seguimiento de flanco
      }
    } else {
      if (echoActive) {
        // Flanco de bajada: el haz acustico acaba de cruzar el blanco por completo
        echoActive = false;
        echoEndAngle = currentServoAngle;
        float centroid = (echoStartAngle + echoEndAngle) / 2.0;
        kf_angle.update(centroid);
      }

      if (reachedTurnaround) {
        // Media oscilacion completada sin ver el blanco
        lostSweepCount++;
        // Si no hay eco, amortiguar de inmediato la velocidad para no divergir
        kf_angle.dampVelocity(0.3);
        kf_dist.dampVelocity(0.5);

        // Ensanchamiento adaptativo del sector para recuperar blancos rapidos
        currentSectorAmp = min(MAX_SECTOR_HALF_AMP, BASE_SECTOR_HALF_AMP + lostSweepCount * 7.0);

        if (lostSweepCount >= MAX_LOST_SWEEPS) {
          currentMode = MODE_SEARCH;
          currentSectorAmp = BASE_SECTOR_HALF_AMP;
          echoActive = false;
          Serial.println(F("<<< Objetivo perdido. Reanudando busqueda..."));
        }
      }
    }
  }

  // --------------------------------------------------------------------------
  // TELEMETRIA SERIE (Regulada a 150 ms para no saturar el buffer UART a 9600)
  // --------------------------------------------------------------------------
  if (now - lastTelemetryMs >= 150) {
    lastTelemetryMs = now;
    if (currentMode == MODE_TRACK_SECTOR) {
      Serial.print(F("[TRK] Dist: "));
      Serial.print(kf_dist.pos, 1);
      Serial.print(F(" cm | ObjAng: "));
      Serial.print((int)round(kf_angle.pos));
      Serial.print(F(" deg (v: "));
      if (kf_angle.vel >= 0) Serial.print('+');
      Serial.print(kf_angle.vel, 1);
      Serial.print(F(") | Servo: "));
      Serial.print((int)round(currentServoAngle));
      Serial.print(F(" deg | Sec: +/-"));
      Serial.print((int)round(currentSectorAmp));
      Serial.println(F(" deg"));
    }
  }
}
