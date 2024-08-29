import serial

ser = serial.Serial("COM5",115200)

buffer = []
latest_message = []
message_length = 8

while True:
    #get the length of buffer
    buffer_size = ser.in_waiting
    buffer = ser.read(size=buffer_size)

    print(buffer_size)
    #get the last message in buffer
    if(buffer_size > message_length):
        i = buffer_size - buffer_size % message_length
        lastest_message = buffer[i-message_length : i]
        print(lastest_message)

    #clear the buffer
    ser.reset_input_buffer()

    input()
    