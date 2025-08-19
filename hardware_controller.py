def move_to_angle(self, target_angle):
    """
    Moves the stepper motor using hardware PWM for large moves and waveforms for small moves.
    This prevents pigpio crashes on moves >180 degrees.
    """
    # Stop any existing hardware PWM before starting
    self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 500000)
    self.pi.wave_tx_stop()
    self.pi.wave_clear()

    # Clamp target angle
    target_angle = max(MotorParams.PAN_MIN_ANGLE, min(MotorParams.PAN_MAX_ANGLE, target_angle))

    print(f"[HWCtrl-Stepper] Moving to {target_angle:.3f}°")

    # Calculate the move
    current_pos_steps = self.step_count
    target_pos_steps = int(target_angle / MICROSTEP_ANGLE)
    error_steps = target_pos_steps - current_pos_steps

    # Find shortest path
    steps_per_rotation = int(360.0 / MICROSTEP_ANGLE)
    if abs(error_steps) > (steps_per_rotation / 2):
        if error_steps > 0:
            error_steps -= steps_per_rotation
        else:
            error_steps += steps_per_rotation

    total_steps = abs(error_steps)

    if total_steps < 1:
        print("[HWCtrl-Stepper] Already at target position.")
        return

    # Set direction
    direction = 1 if error_steps > 0 else 0
    self.pi.write(STEPPER_DIR_PIN, direction)
    print(f"[HWCtrl-Stepper] Steps to move: {total_steps}, Direction: {'CW' if direction else 'CCW'}")

    # CRITICAL: Use different methods based on move size
    # Threshold for waveform vs PWM (adjust as needed)
    WAVEFORM_THRESHOLD = 1600  # About 90 degrees of movement

    if total_steps > WAVEFORM_THRESHOLD:
        # Use hardware PWM for large moves
        self._move_with_pwm(total_steps, direction)
    else:
        # Use waveform for small, precise moves
        self._move_with_waveform(total_steps, direction)

    # Log final position
    final_pos_deg = self.shared_data["stepper_degrees"].value
    print(f"[HWCtrl-Stepper] Movement complete. Final position: {final_pos_deg:.3f}°")


def _move_with_pwm(self, total_steps, direction):
    """
    Execute large moves using hardware PWM with acceleration/deceleration.
    More reliable for moves >90 degrees.
    """
    print(f"[HWCtrl-Stepper] Using PWM mode for {total_steps} steps")

    # Calculate acceleration parameters
    accel_steps = min(MotorParams.ACCEL_STEPS, total_steps // 3)
    decel_steps = min(MotorParams.ACCEL_STEPS, total_steps // 3)
    cruise_steps = total_steps - accel_steps - decel_steps

    steps_done = 0

    # Acceleration phase
    print("[HWCtrl-Stepper] Accelerating...")
    for i in range(1, accel_steps + 1):
        if not self.running:
            break
        speed = MotorParams.STEPPER_MIN_SPEED + (
                (MotorParams.STEPPER_MAX_SPEED - MotorParams.STEPPER_MIN_SPEED) *
                (i / accel_steps)
        )
        freq = min(int(speed), MotorParams.STEPPER_MAX_SPEED)
        self.pi.hardware_PWM(STEPPER_PULSE_PIN, freq, 500000)

        # Calculate time for this speed segment
        segment_time = 1.0 / speed
        time.sleep(segment_time)
        steps_done += 1

    # Cruise phase
    if cruise_steps > 0 and self.running:
        print(f"[HWCtrl-Stepper] Cruising for {cruise_steps} steps...")
        cruise_freq = min(MotorParams.STEPPER_MAX_SPEED, MotorParams.STEPPER_CRUISE_SPEED)
        self.pi.hardware_PWM(STEPPER_PULSE_PIN, cruise_freq, 500000)

        cruise_time = cruise_steps / cruise_freq
        start_cruise = time.time()

        # Wait for cruise phase with periodic checks
        while (time.time() - start_cruise) < cruise_time and self.running:
            time.sleep(0.01)
        steps_done += cruise_steps

    # Deceleration phase
    print("[HWCtrl-Stepper] Decelerating...")
    for i in range(decel_steps, 0, -1):
        if not self.running:
            break
        speed = MotorParams.STEPPER_MIN_SPEED + (
                (MotorParams.STEPPER_MAX_SPEED - MotorParams.STEPPER_MIN_SPEED) *
                (i / decel_steps)
        )
        freq = min(int(speed), MotorParams.STEPPER_MAX_SPEED)
        self.pi.hardware_PWM(STEPPER_PULSE_PIN, freq, 500000)

        segment_time = 1.0 / speed
        time.sleep(segment_time)
        steps_done += 1

    # Stop PWM
    self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 500000)
    print(f"[HWCtrl-Stepper] PWM move complete. Steps executed: {steps_done}")


def _move_with_waveform(self, total_steps, direction):
    """
    Execute small moves using pigpio waveforms for precise control.
    Best for moves <90 degrees.
    """
    print(f"[HWCtrl-Stepper] Using waveform mode for {total_steps} steps")

    # Calculate acceleration/deceleration steps
    if total_steps <= MotorParams.ACCEL_STEPS * 2:
        accel_steps_actual = total_steps // 2
        decel_steps_actual = total_steps - accel_steps_actual
    else:
        accel_steps_actual = MotorParams.ACCEL_STEPS
        decel_steps_actual = MotorParams.ACCEL_STEPS

    # Build the waveform
    pulses = []

    # Limit check to prevent excessive pulses
    MAX_PULSES = 3000  # pigpio limit is around 3000-4000 pulses
    if total_steps * 2 > MAX_PULSES:
        print(f"[HWCtrl-Stepper] Warning: Move too large for waveform ({total_steps * 2} pulses)")
        # Fall back to PWM
        self._move_with_pwm(total_steps, direction)
        return

    # Acceleration
    for i in range(1, accel_steps_actual + 1):
        speed = MotorParams.STEPPER_MIN_SPEED + (
                (MotorParams.STEPPER_MAX_SPEED - MotorParams.STEPPER_MIN_SPEED) *
                (i / accel_steps_actual)
        )
        delay_us = int(500000 / speed)
        pulses.append(pigpio.pulse(1 << STEPPER_PULSE_PIN, 0, delay_us))
        pulses.append(pigpio.pulse(0, 1 << STEPPER_PULSE_PIN, delay_us))

    # Cruise
    cruise_steps = total_steps - (accel_steps_actual + decel_steps_actual)
    if cruise_steps > 0:
        delay_us = int(500000 / MotorParams.STEPPER_MAX_SPEED)
        for _ in range(cruise_steps):
            pulses.append(pigpio.pulse(1 << STEPPER_PULSE_PIN, 0, delay_us))
            pulses.append(pigpio.pulse(0, 1 << STEPPER_PULSE_PIN, delay_us))

    # Deceleration
    for i in range(decel_steps_actual, 0, -1):
        speed = MotorParams.STEPPER_MIN_SPEED + (
                (MotorParams.STEPPER_MAX_SPEED - MotorParams.STEPPER_MIN_SPEED) *
                (i / decel_steps_actual)
        )
        delay_us = int(500000 / speed)
        pulses.append(pigpio.pulse(1 << STEPPER_PULSE_PIN, 0, delay_us))
        pulses.append(pigpio.pulse(0, 1 << STEPPER_PULSE_PIN, delay_us))

    # Add wave to pigpio
    try:
        self.pi.wave_add_generic(pulses)
        wave_id = self.pi.wave_create()

        if wave_id >= 0:
            print(f"[HWCtrl-Stepper] Sending wave {wave_id} with {len(pulses) // 2} pulses")
            self.pi.wave_send_once(wave_id)

            # Wait for completion
            while self.pi.wave_tx_busy() and self.running:
                time.sleep(0.01)

            # Clean up
            self.pi.wave_delete(wave_id)
            print("[HWCtrl-Stepper] Wave complete")
        else:
            print(f"[HWCtrl-Stepper] Wave creation failed (id={wave_id}), falling back to PWM")
            self._move_with_pwm(total_steps, direction)

    except Exception as e:
        print(f"[HWCtrl-Stepper] Waveform error: {e}, falling back to PWM")
        self._move_with_pwm(total_steps, direction)