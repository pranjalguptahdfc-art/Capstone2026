"""
gas_reader.py
-------------
Ethylene gas sensor reader.

For demo: simulates a reading.
For real hardware: reads from ESP32 serial port.
"""

import time
import random
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Thresholds
ETHYLENE_LOW = 0.5       # ppm - no significant ripening
ETHYLENE_HIGH = 2.0      # ppm - active ripening
ETHYLENE_CRITICAL = 5.0  # ppm - near spoilage


class GasReader:
    """
    Reads ethylene gas sensor values.

    In demo mode (default), simulates readings.
    When connected to ESP32, reads from serial port.
    """

    def __init__(self, demo=True, serial_port=None, baudrate=115200):
        self.demo = demo
        self.serial_port = serial_port
        self.baudrate = baudrate
        self._serial = None

        if not demo and serial_port:
            self._connect_serial()

    def _connect_serial(self):
        try:
            import serial
            self._serial = serial.Serial(self.serial_port, self.baudrate, timeout=1)
            logger.info(f"Connected to gas sensor on {self.serial_port}")
        except ImportError:
            logger.warning("pyserial not installed - falling back to demo mode")
            self.demo = True
        except Exception as e:
            logger.warning(f"Serial connection failed: {e} - falling back to demo mode")
            self.demo = True

    def read(self):
        """
        Read the current gas sensor value.

        Returns dict with:
            ethylene_ppm, raw_adc, status (LOW/MODERATE/HIGH/CRITICAL), timestamp
        """
        if self.demo:
            return self._read_demo()
        return self._read_serial()

    def _read_demo(self):
        if random.random() < 0.7:
            ppm = round(random.uniform(0.0, ETHYLENE_LOW), 3)
        elif random.random() < 0.8:
            ppm = round(random.uniform(ETHYLENE_LOW, ETHYLENE_HIGH), 3)
        else:
            ppm = round(random.uniform(ETHYLENE_HIGH, ETHYLENE_CRITICAL), 3)
        raw_adc = int(ppm / 10.0 * 4095)
        status = self._classify(ppm)
        return {
            "ethylene_ppm": ppm,
            "raw_adc": raw_adc,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _read_serial(self):
        if self._serial is None or not self._serial.is_open:
            return self._read_demo()
        try:
            line = self._serial.readline().decode("utf-8").strip()
            parts = line.split(",")
            if len(parts) >= 2:
                ppm = float(parts[0])
                raw_adc = int(parts[1])
            else:
                ppm = float(parts[0])
                raw_adc = int(ppm / 10.0 * 4095)
            status = self._classify(ppm)
            return {
                "ethylene_ppm": ppm,
                "raw_adc": raw_adc,
                "status": status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Serial read error: {e}")
            return self._read_demo()

    @staticmethod
    def _classify(ppm):
        if ppm < ETHYLENE_LOW:
            return "LOW"
        elif ppm < ETHYLENE_HIGH:
            return "MODERATE"
        elif ppm < ETHYLENE_CRITICAL:
            return "HIGH"
        else:
            return "CRITICAL"

    def close(self):
        if self._serial and self._serial.is_open:
            self._serial.close()
            logger.info("Gas sensor serial connection closed")
