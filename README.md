# Generic Open-source Recorder for Data Acquisition and Time-frequency Analysis

1. Download and flash the [latest Raspberry Pi OS](https://www.raspberrypi.com/software/).
2. Change the following settings in config.txt on boot partition.

    > hdmi_group =2
    > hdmi_mode=87
    > hdmi_cvt=1024 600 60 
    > arm_freq=21751
    > dtparam=i2c_arm=on,i2c_arm_baudrate=1800000 
    > #arm_64bit=1

3. Install RealVNC Server.
    >  sudo apt install realvnc-vnc-server

4. Install the  [latest `python` distribution](https://allurcode.com/install-latest-version-of-python-on-raspberry-pi/).   

5. Clone repository with  
    > git clone https://github.com/nogleo/gordata.git 