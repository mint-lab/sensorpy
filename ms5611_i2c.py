#!/usr/bin/python3

# TODO: Class로 변경
# TODO: Windows에서도 호환 가능하도록 수정
import time
from smbus2 import SMBus

# Debugging
DEBUG = False

# MS5637 I2C address
MS5637_ADDR = 0x76

# MS5637 commands
RESET_COMMAND = 0x1E
PROM_READ_COMMAND = 0xA0
CONVERT_D1_COMMAND = 0x40
CONVERT_D2_COMMAND = 0x50
ADC_READ_COMMAND = 0x00

# Initialize I2C bus
bus = SMBus(22)


def reset_sensor():
    bus.write_byte(MS5637_ADDR, RESET_COMMAND)
    if DEBUG:
        print("Sensor reset")
    return True


def read_prom():
    prom_data = []
    for i in range(1, 7):
        prom_cmd = PROM_READ_COMMAND + (i * 2)
        data = bus.read_i2c_block_data(MS5637_ADDR, prom_cmd, 2)
        if DEBUG:
            print(f"PROM data: {data}")
        prom_data.append((data[0] << 8) + data[1])

    if DEBUG:
        print(f"PROM data: {prom_data}")
    return prom_data


def initialize():
    reset_sensor()
    prom_data = read_prom()
    return prom_data


def read_adc(command):
    bus.write_byte(MS5637_ADDR, command)
    time.sleep(0.2)  # Conversion time
    data = bus.read_i2c_block_data(MS5637_ADDR, ADC_READ_COMMAND, 3)
    adc_data = (data[0] << 16) + (data[1] << 8) + data[2]

    if DEBUG:
        print(f"ADC data: {adc_data}")
    return adc_data


def read_sensor():
    # Initialize sensor
    prom_data = initialize()

    # Start pressure conversion
    D1 = read_adc(CONVERT_D1_COMMAND)

    # Start temperature conversion
    D2 = read_adc(CONVERT_D2_COMMAND)

    # Calculate temperature and pressure using calibration data
    dT = D2 - (prom_data[4] * 256)
    TEMP = 2000 + dT * prom_data[5] / 8388608

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

    OFF = (prom_data[1] * 131072) + ((prom_data[3] * dT) / 64)
    OFF -= OFF2
    SENS = (prom_data[0] * 65536) + ((prom_data[2] * dT) / 128)
    SENS -= SENS2

    P = ((D1 * SENS / 2097152) - OFF) / 32768

    return (TEMP - T2) / 100, P / 100


if __name__ == "__main__":
    while True:
        # Read sensor data
        try:
            temperature, pressure = read_sensor()
            print(f"Temperature: {temperature:.2f} Celsius")
            print(f"Pressure: {pressure:.2f} hPa")
        except Exception as e:
            print(f"Error: {e}")
