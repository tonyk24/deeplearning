select * from device_measurement_value where idDevice = 215 into outfile '/var/tmp/output_outdoor_data.csv' fields enclosed by '"' terminated by ';' escaped by '"' lines terminated by '\r\n';
