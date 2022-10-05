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

enum udp_direction {UDP_SEND = 0,    UDP_RECV = 1};
enum udp_mode      {UDP_UNICAST = 0, UDP_MULTICAST=1, UDP_BROADCAST=2};

//#define UDP_DEFAULT_MODE      UDP_UNICAST
//#define UDP_DEFAULT_DIRECTION UDP_SEND

/*! A function to create udp socket for different mode (unicast, broadcast and multicast) and two directions (send and receive)
  
 * @param[in] ip        IP address, 0.0.0.0 is INADDR_ANY, 255.255.255.255 is INADDR_UDP_BROADCAST
 * @param[in] group     multicast group, only be used when mode == UDP_MULTICAST
 * @param[in] port      port number 
 * @param[in] reuse     reuse the interface if it is nonzero
 * @param[in] bufsz     socket buffer size in MBytes, 0 or negative value means the default value will be used 
 * @param[in] tout      time out in seconds, 0 means the socket is nonblock, negative means block without timeout
 * @param[in] mode      socket mode, which can be UDP_UNICAST, UDP_MULTICAST or UDP_BROADCAST
 * @param[in] direction which direction the traffic will be, send or receive

 * @param[out] sock     to return create socket
 */
int create_udp_socket(char *ip, char *group, int port, int &sock,
		      int reuse, int bufsz, double tout,
		      enum udp_mode mode, enum udp_direction direction);
#endif
