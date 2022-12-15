#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "tcp_utils.h"

int create_tcp_socket(char *ip, int port, int &sock,
		      int reuse, int bufsz, double tout, int depth,
		      enum tcp_direction direction){

  int sock0 = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

  // Setup reuse if it is required
  if(reuse){
    if (setsockopt(sock0, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse))){
      fprintf(stderr, "CREATE_TCP_SOCKET_ERROR: Could not enable SO_REUSEADDR to %s_%d, "
	      "which happens at \"%s\", line [%d], has to abort.\n",
	      ip, port, __FILE__, __LINE__);
      
      close(sock0);
      return EXIT_FAILURE;
    }    
  }

  // Setup socket according tout
  // tout == 0, NONBLOCK
  // tout > 0, block with timeout
  // tout < 0, block without timeout 
  if (tout == 0){
    if(fcntl(sock0, F_SETFL, fcntl(sock0, F_GETFL, 0) | O_NONBLOCK) == -1){
      fprintf(stderr, "CREATE_TCP_SOCKET_ERROR: Could not setup NONBLOCK to %s_%d, "
	      "which happens at \"%s\", line [%d], has to abort.\n",
	      ip, port, __FILE__, __LINE__);
      
      close(sock0);
      return EXIT_FAILURE;      
    }    
  }  
  else if(tout > 0){
    int tout_flag = 0;
    
    if(direction == TCP_RECV){
      tout_flag = SO_RCVTIMEO;
    }
    else{    
      tout_flag = SO_SNDTIMEO;
    }
    
    time_t tout_second = tout;
    time_t tout_microsecond = (tout-tout_second)*1E6;
    
    struct timeval timeval_tout = {tout_second, tout_microsecond};
    if (setsockopt(sock0, SOL_SOCKET, tout_flag, (const void*)&timeval_tout, sizeof(timeval_tout))){
      fprintf(stderr, "CREATE_TCP_SOCKET_ERROR: Could not setup TIMEOUT to %s_%d, "
	      "which happens at \"%s\", line [%d], has to abort.\n",
	      ip, port, __FILE__, __LINE__);
      
      close(sock0);
      return EXIT_FAILURE;
    }
  }

  // Setup bufsz if it is > 0
  if(bufsz > 0){
    int buf_flag  = 0;
  
    if(direction==TCP_RECV){
      buf_flag  = SO_RCVBUF;
    }
    else{ 
      buf_flag  = SO_SNDBUF;
    }

    int nbyte_buffer = bufsz*1E6;
    if (setsockopt(sock0, SOL_SOCKET, buf_flag, &nbyte_buffer, sizeof(nbyte_buffer))) {
      fprintf(stderr, "CREATE_TCP_SOCKET_ERROR: Could not set socket BUF to %s_%d, "
	      "which happens at \"%s\", line [%d], has to abort.\n",
	      ip, port, __FILE__, __LINE__);
      
      close(sock0);
      return EXIT_FAILURE;
    }
  }

  struct sockaddr_in sa = {0};
  sa.sin_port   = htons(port);
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = inet_addr(ip);
  
  if(direction == TCP_SEND){
    /* Set the linger option so that if we need to send a message and
       close the socket, the message shouldn't get lost */
    struct linger linger = {1, 1};
    if (setsockopt(sock0, SOL_SOCKET, SO_LINGER, (char *)&linger, sizeof(struct linger))!=0) {
      close(sock0);
      sock0 = 0;
      return EXIT_FAILURE;
    }
    
    if (connect(sock0, (struct sockaddr *)&sa, sizeof(sa))){
      fprintf(stderr, "CREATE_TCP_SOCKET_ERROR: Can not connect to %s_%d, "
	      "which happens at \"%s\", line [%d], has to abort.\n",
	      ip, port, __FILE__, __LINE__);
      perror("connect");
      
      close(sock0);
      return EXIT_FAILURE;
    } // if (connect(sock0, (struct sockaddr *)&sa, sizeof(sa))){

    sock = sock0;
  } // if(direction == TCP_SEND)
  else{
    /* receive */
    if (bind(sock0, (struct sockaddr *) &sa, sizeof(sa)) < 0) {        
      fprintf(stderr, "CREATE_TCP_SOCKET_ERROR: Could not bind to %s_%d, "
	      "which happens at \"%s\", line [%d], has to abort.\n",
	      ip, port, __FILE__, __LINE__);
      
      close(sock0);
      return EXIT_FAILURE;
    } // if (bind(sock0, (struct sockaddr *) &sa, sizeof(sa)) < 0) {
    
    /*listen*/
    if(depth <= 0){
      depth = TCP_DEFAULT_BACKLOG;
    }  // depth <= 0
    if (listen(sock0, depth) < 0){        
      fprintf(stderr, "CREATE_TCP_SOCKET_ERROR: Could not listen to a given TCP socket, "
	      "which happens at \"%s\", line [%d], has to abort.\n",
	      __FILE__, __LINE__);
      
      close(sock0);
      return EXIT_FAILURE;
    } // listen < 0
    
    /*Accpet*/
    struct sockaddr_in cli_addr;
    socklen_t cli_len = sizeof(cli_addr);
    sock = accept(sock0, (struct sockaddr*)&cli_addr, &cli_len);
    if (sock < 0){          
      fprintf(stderr, "CREATE_TCP_SOCKET_ERROR: Could not accept a given TCP socket, "
	      "which happens at \"%s\", line [%d], has to abort.\n",
	      __FILE__, __LINE__);
      
      close(sock0);
      close(sock);
      return EXIT_FAILURE;
    } // sock < 0
  }  // else
    
  return EXIT_SUCCESS;
}

int sendbuf_tcp(int sock, char *buf, int nbytes) {
  int nwrote;
  int ntowrite = nbytes;
  char *ptr = buf;
  
  while (ntowrite>0) {
    nwrote = send(sock, ptr, ntowrite, 0);
    if (nwrote==-1) {
      if (errno == EINTR)
	continue;
      fprintf(stderr, "SENDBUF_TCP_ERROR: Error writing to network with EINTR, "
	      "which happens at \"%s\", line [%d], has to abort.\n",
	      __FILE__, __LINE__);
      return EXIT_FAILURE;

    } else if (nwrote==0) {
      fprintf(stderr, "SENDBUF_TCP_WARN: Did not write any bytes (0 bytes) "
	      "which happens at \"%s\", line [%d], has to abort.\n",
	      __FILE__, __LINE__);
      
      return EXIT_FAILURE;
    } else {
      ntowrite -= nwrote;
      ptr += nwrote;
    }
  }
  
  return EXIT_SUCCESS;
}
