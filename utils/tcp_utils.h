#ifndef _TCP_UTILS_H
#define _TCP_UTILS_H

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

enum tcp_direction {TCP_SEND = 0, TCP_RECV = 1};
#define TCP_DEFAULT_BACKLOG 4096

/*! A function to create tcp socket for different mode (unicast, broadcast and multicast) and two directions (send and receive)
  
 * @param[in] ip        IP address, 0.0.0.0 is INADDR_ANY, 255.255.255.255 is INADDR_TCP_BROADCAST, for sender use NULL will not bind socket to a physical interface
 * @param[in] port      port number 
 * @param[in] reuse     reuse the interface if it is nonzero
 * @param[in] bufsz     socket buffer size in MBytes, 0 or negative value means the default value will be used 
 * @param[in] tout      time out in seconds, 0 means the socket is nonblock, negative means block without timeout
 * @param[in] depth     Depth of listen option, 0 or negative number apply default value 4096
 * @param[in] direction which direction the traffic will be, send or receive

 * @param[out] sock     to return create socket
 */
int create_tcp_socket(char *ip, int port, int &sock,
		      int reuse, int bufsz, double tout, int depth,
		      enum tcp_direction direction);
#endif
