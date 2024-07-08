import time
from smbus2 import SMBus


# TODO: Windows에서도 호환 가능하도록 수정
# TODO: 다른 barometer 추가
class MS5637:
    def __init__(self, bus: SMBus, i2c_address: int, verbose=False) -> None:
        """
        @brief: Initialize the MS5637 sensor
        @param bus: SMBus object
        @param i2c_address: I2C address of the sensor
        @param verbose: Enable debug output
        """
        self.bus = bus
        self.i2c_address = i2c_address
        self.verbose = verbose

        self._RESET_COMMAND = 0x1E
        self._PROM_READ_COMMAND = 0xA0
        self._CONVERT_D1_COMMAND = 0x40
        self._CONVERT_D2_COMMAND = 0x50
        self._ADC_READ_COMMAND = 0x00

        if not self.reset_sensor():
            raise Exception("Sensor not found")
        else:
            self.prom_data = self.read_prom()
            print("Sensor initialized")

    def check_i2c_device(self) -> bool:
        """
        @brief: Check if the sensor is connected
        @return: True if the sensor is connected
        """
        try:
            self.bus.read_byte(self.i2c_address)
            return True
        except Exception as e:
            if self.verbose:
                print(f"[MS5637] {e}")
            return False

    def reset_sensor(self) -> bool:
        """
        @brief: Reset the sensor
        @return: True if successful
        """
        if self.check_i2c_device():
            self.bus.write_byte(self.i2c_address, self._RESET_COMMAND)
            if self.verbose:
                print("Sensor reset")
            return True
        return False

    def read_prom(self) -> list:
        """
        @brief: Read PROM data from the sensor
        @return: List of PROM data
        """
        prom_data = []
        for i in range(1, 7):
            prom_cmd = self._PROM_READ_COMMAND + (i * 2)
            data = self.bus.read_i2c_block_data(self.i2c_address, prom_cmd, 2)
            prom_data.append((data[0] << 8) + data[1])

        if self.verbose:
            print(f"PROM data: {prom_data}")
        return prom_data

    def read_adc(self, command: int) -> int:
        """
        @brief: Read ADC data from the sensor
        @param command: ADC command
        @return: ADC data
        """
        self.bus.write_byte(self.i2c_address, command)
        time.sleep(0.2)
        data = self.bus.read_i2c_block_data(self.i2c_address, self._ADC_READ_COMMAND, 3)
        adc_data = (data[0] << 16) + (data[1] << 8) + data[2]

        if self.verbose:
            print(f"ADC data: {adc_data}")
        return adc_data

    def read_sensor(self) -> tuple:
        """
        @brief: Read sensor data
        @return: Pressure and temperature data
        """
        D1 = self.read_adc(self._CONVERT_D1_COMMAND)  # Pressure
        D2 = self.read_adc(self._CONVERT_D2_COMMAND)  # Temperature

        # Calculate pressure and temperature using calibration data
        dT = D2 - self.prom_data[4] * 256
        TEMP = 2000 + dT * self.prom_data[5] / 8388608

        # Second order temperature compensation
        if TEMP < 2000:
            T2 = 3 * (dT * dT) / 8589934592
            OFF2 = 61 * ((TEMP - 2000) * (TEMP - 2000)) / 16
            SENS2 = 29 * ((TEMP - 2000) * (TEMP - 2000)) / 16

            if TEMP < -1500:
                OFF2 += 17 * ((TEMP + 1500) * (TEMP + 1500))
                SENS2 += 9 * ((TEMP + 1500) * (TEMP + 1500))
        else:
            T2 = 5 * (dT * dT) / 274877906944
            OFF2 = 0
            SENS2 = 0

        OFF = (self.prom_data[1] * 131072) + ((self.prom_data[3] * dT) / 64)
        OFF -= OFF2
        SENS = (self.prom_data[0] * 65536) + ((self.prom_data[2] * dT) / 128)
        SENS -= SENS2

        P = ((D1 * SENS / 2097152) - OFF) / 32768
        return P / 100, (TEMP - T2) / 100


def test():
    bus_number = 88  # MINT Lab udev rules in mint_cart_ros
    device_address = 0x76

    with SMBus(bus_number) as bus:
        sensor = MS5637(bus, device_address, verbose=True)
        pressure, temperature = sensor.read_sensor()
        print(f"Pressure: {pressure} hPa, Temperature: {temperature} °C")


if __name__ == "__main__":
    test()
