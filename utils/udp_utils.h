#ifndef _UDP_UTILS_H
#define _UDP_UTILS_H

#include <stdlib.h>
#include <assert.h>
#include <limits.h>

#include <time.h>
#include <getopt.h>

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/socket.h>
#include <netinet/ip.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>

enum udp_direction {SEND = 0,    RECV = 1};
enum udp_mode      {UNICAST = 0, MULTICAST=1, BROADCAST=2};

#define UDP_DEFAULT_MODE      UNICAST
#define UDP_DEFAULT_DIRECTION SEND

int create_udp_socket(char *ip, int port, char *ip_src, int port_src, 
		      int &sock, int reuse, int window, double tout,
		      enum udp_mode mode, enum udp_direction direction);
#endif
