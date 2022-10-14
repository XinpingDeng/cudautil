#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "udp_utils.h"

// INADDR_ANY (0.0.0.0) and INADDR_UDP_BROADCAST (255.255.255.255)

int create_udp_socket(char *ip, char *group, int port, int &sock,
		      int reuse, int bufsz, double tout,
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
      fprintf(stderr, "CREATE_UDP_SOCKET_ERROR:\tCould not setup NONBLOCK to %s_%d, "
	      "which happens at \"%s\", line [%d], has to abort.\n",
	      ip, port, __FILE__, __LINE__);
      
      close(sock);
      return EXIT_FAILURE;      
    }    
  }  
  else if(tout > 0){
    int tout_flag = 0;
    
    if(direction == UDP_RECV){
      tout_flag = SO_RCVTIMEO;
    }
    else{    
      tout_flag = SO_SNDTIMEO;
    }
    
    time_t tout_second = tout;
    time_t tout_microsecond = (tout-tout_second)*1E6;
    
    struct timeval timeval_tout = {tout_second, tout_microsecond};
    if (setsockopt(sock, SOL_SOCKET, tout_flag, (const void*)&timeval_tout, sizeof(timeval_tout))){
      fprintf(stderr, "CREATE_UDP_SOCKET_ERROR:\tCould not setup TIMEOUT to %s_%d, "
	      "which happens at \"%s\", line [%d], has to abort.\n",
	      ip, port, __FILE__, __LINE__);
      
      close(sock);
      return EXIT_FAILURE;
    }
  }

  // Setup bufsz if it is > 0
  if(bufsz > 0){
    int buf_flag  = 0;
  
    if(direction==UDP_RECV){
      buf_flag  = SO_RCVBUF;
    }
    else{ 
      buf_flag  = SO_SNDBUF;
    }

    int nbyte_buffer = bufsz*1E6;
    if (setsockopt(sock, SOL_SOCKET, buf_flag, &nbyte_buffer, sizeof(nbyte_buffer))) {
      fprintf(stderr, "CREATE_UDP_SOCKET_ERROR:\tCould not set socket BUF to %s_%d, "
	      "which happens at \"%s\", line [%d], has to abort.\n",
	      ip, port, __FILE__, __LINE__);
      
      close(sock);
      return EXIT_FAILURE;
    }
  }

  struct sockaddr_in sa = {0};
  sa.sin_family = AF_INET;
    
  if(direction == UDP_SEND){    
    // In send direction, UDP_BROADCAST is different from others
    if(mode == UDP_BROADCAST){
      int broadcast = 1;
      if (setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast))){
	fprintf(stderr, "CREATE_UDP_SOCKET_ERROR:\tCould not setup SO_BROADCAST to %s_%d, "
		"which happens at \"%s\", line [%d], has to abort.\n",
		ip, port, __FILE__, __LINE__);
	
	close(sock);
	return EXIT_FAILURE;
      } // if (setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast)))
    } // if(mode == UDP_BROADCAST)

    if(ip != NULL){
      // if send ip is INADDR_ANY, it connects to loopback,
      // with which we can not use it to send data to a remote machine
      // use NULL to let OS decide ip and port for sending
      sa.sin_port = htons(port);
      sa.sin_addr.s_addr = inet_addr(ip);
      if (connect(sock, (struct sockaddr *)&sa, sizeof(sa))){
	fprintf(stderr, "CREATE_UDP_SOCKET_ERROR:\tCan not bind to %s_%d, "
		"which happens at \"%s\", line [%d], has to abort.\n",
		ip, port, __FILE__, __LINE__);
	
	close(sock);
	return EXIT_FAILURE;
      } // if (connect(sock, (struct sockaddr *)&sa, sizeof(sa))){
    } // if(ip != NULL)
  } // if(direction == UDP_SEND)
  else{
    // Receive ip can be INADDR_ANY
    sa.sin_port   = htons(port);
    if(ip == NULL){
      sa.sin_addr.s_addr = htonl(INADDR_ANY);
    }
    else{
      sa.sin_addr.s_addr = inet_addr(ip);
    }
    /* receive */
    if (bind(sock, (struct sockaddr *) &sa, sizeof(sa)) < 0) {        
      fprintf(stderr, "CREATE_UDP_SOCKET_ERROR:\tCould not bind to %s_%d, "
	      "which happens at \"%s\", line [%d], has to abort.\n",
	      ip, port, __FILE__, __LINE__);
      
      close(sock);
      return EXIT_FAILURE;
    } // if (bind(sock, (struct sockaddr *) &sa, sizeof(sa)) < 0) {
    
    if(mode == UDP_MULTICAST){
      struct ip_mreq mreq = {0};
      mreq.imr_multiaddr.s_addr = inet_addr(group);
      if(ip == NULL){
	mreq.imr_interface.s_addr =  htonl(INADDR_ANY);
      }
      else{
	mreq.imr_interface.s_addr = inet_addr(ip);
      }
      
      if (setsockopt(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP,
		     &mreq, sizeof(mreq)) < 0) {
	fprintf(stderr, "CREATE_UDP_SOCKET_ERROR:\tCould not add to multicast group %s, "
		"which happens at \"%s\", line [%d], has to abort.\n",
		group, __FILE__, __LINE__);
      
	close(sock);

	return EXIT_FAILURE;
      } // if (setsockopt(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP,
    } // if(mode == UDP_MULTICAST){
  } //else 

  return EXIT_SUCCESS;
}
