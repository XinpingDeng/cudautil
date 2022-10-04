#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "udp_utils.h"

int create_udp_socket(char *ip, int port, char *ip_src, int port_src, 
		      int &sock, int reuse, int window, double tout,
		      enum udp_mode mode, enum udp_direction direction){

  sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);

  // Setup reuse if it is required
  if(reuse){
    if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse))){
      fprintf(stderr, "CREATE_UDP_SOCKET_ERROR:\tCould not enable SO_REUSEADDR to %s_%d, "
	      "which happens at \"%s\", line [%d], has to abort.\n",
	      ip, port, __FILE__, __LINE__);
      
      close(sock);
      return EXIT_FAILURE;
    }    
  }

  // Setup socket according tout
  // tout == 0, NONBLOCK
  // tout > 0, block with timeout
  // tout < 0, block without timeout 
  if (tout == 0){
    if(fcntl(sock, F_SETFL, fcntl(sock, F_GETFL, 0) | O_NONBLOCK) == -1){
      fprintf(stderr, "CREATE_UDP_SOCKET_ERROR:\tCould not setup NONBLOCK to socket, "
	      "which happens at \"%s\", line [%d], has to abort.\n",
	      __FILE__, __LINE__);
      
      close(sock);
      return EXIT_FAILURE;      
    }    
  }  
  else if(tout > 0){
    int tout_flag = 0;
    
    if(direction == RECV){
      tout_flag = SO_RCVTIMEO;
    }
    else{    
      tout_flag = SO_SNDTIMEO;
    }
    
    time_t tout_second = tout;
    time_t tout_microsecond = (tout-tout_second)*1E6;
    
    struct timeval timeval_tout = {tout_second, tout_microsecond};
    if (setsockopt(sock, SOL_SOCKET, tout_flag, (const void*)&timeval_tout, sizeof(timeval_tout))){
      fprintf(stderr, "CREATE_UDP_SOCKET_ERROR:\tCould not setup TIMEOUT to socket, "
	      "which happens at \"%s\", line [%d], has to abort.\n",
	      __FILE__, __LINE__);
      
      close(sock);
      return EXIT_FAILURE;
    }
  }

  // Setup window if it is not 0
  if(window){
    int buf_flag  = 0;
  
    if(recv){
      buf_flag  = SO_RCVBUF;
    }
    else{ 
      buf_flag  = SO_SNDBUF;
    }

    int nbyte_window = window*1E6;
    if (setsockopt(sock, SOL_SOCKET, buf_flag, &nbyte_window, sizeof(nbyte_window))) {
      fprintf(stderr, "CREATE_UDP_SOCKET_ERROR:\tCould not set socket BUF to %s_%d, "
	      "which happens at \"%s\", line [%d], has to abort.\n",
	      ip, port, __FILE__, __LINE__);
      
      close(sock);
      return EXIT_FAILURE;
    }
  }

  struct sockaddr_in sa = {0};
  sa.sin_family      = AF_INET;
  sa.sin_port        = htons(port);

  if(direction == SEND){
    sa.sin_addr.s_addr = inet_addr(ip);

    // In send direction, BROADCAST is different from others
    if(mode == BROADCAST){
      int broadcast = 1;
      if (setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast))){
	fprintf(stderr, "BROADCAST_ERROR:\tCould not setup SO_BROADCAST to %s_%d, "
		"which happens at \"%s\", line [%d], has to abort.\n",
		ip, port, __FILE__, __LINE__);
	
	close(sock);
	return EXIT_FAILURE;
      }      
    }
  }
  else{
    if(ip == NULL){
      sa.sin_addr.s_addr = htonl(INADDR_ANY);
    }
    else{
      sa.sin_addr.s_addr = inet_addr(ip);
    }      
    
    /* receive */
    if (bind(sock, (struct sockaddr *) &sa, sizeof(sa)) < 0) {        
      fprintf(stderr, "BROADCAST_ERROR:\tCould not bind to broadcast socket, "
	      "which happens at \"%s\", line [%d], has to abort.\n",
	      __FILE__, __LINE__);
      
      close(sock);
      return EXIT_FAILURE;
    }

    if(mode == MULTICAST){
      struct ip_mreq mreq = {0};
      mreq.imr_multiaddr.s_addr = inet_addr(ip);
      if(ip == NULL){
	mreq.imr_interface.s_addr = htonl(INADDR_ANY);
      }
      else{
	mreq.imr_interface.s_addr = inet_addr(ip);
      }
      
      if (setsockopt(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP,
		     &mreq, sizeof(mreq)) < 0) {
	fprintf(stderr, "MULTICAST_ERROR:\tCould not add to multicast group, "
		"which happens at \"%s\", line [%d], has to abort.\n",
		__FILE__, __LINE__);
      
	close(sock);

	return EXIT_FAILURE;
      }

    }
  }

  return EXIT_SUCCESS;
}
