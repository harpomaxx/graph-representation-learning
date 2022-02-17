Each capture of the CTU-13 was generated using [ARGUS](https://openargus.org/) tool


Variable|Description
---------|-------------
StartTime| datetime of the begining of the conection
Dur| Duration of the connection
Proto| Protocol (eg. TCP, UDP, ICMP, etc.)
SrcAddr| Source IP adresse
Sport|   Source Port
Dir|     direction of the conection (<- or -> or <->)
DstAddr| Destination IP address
Dport| Destion Port
State| State of the connection (see netflow argus reference)
sTos|  type of service field for source IP address
dTos|  type of service field for destionation IP address.
TotPkts| Total packets transfered of the connection
TotBytes| Total bytes transfered in the connection
SrcBytes| Bytes transmited by the source IP Address
Label| Label of the connection
