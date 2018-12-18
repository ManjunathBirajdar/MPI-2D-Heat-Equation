#include <mpi.h>
#include <math.h>
#include <iostream>
#include <chrono>

using namespace std;
#ifdef __cplusplus
extern "C" {
#endif

  int check2DHeat(double** H, long n, long rank, long P, long k); //this assumes array of array and grid block decomposition

#ifdef __cplusplus
}
#endif

/***********************************************
 *         NOTES on check2DHeat.
 ***********************************************
 *         
 
 *
 *************/


// Use similarily as the genA, genx from matmult assignment.
double genH0(long row, long col, long n) {
  double val = (double)(col == (n/2));
  return val;
}

int main(int argc, char* argv[]) {

  if (argc < 3) {
    std::cerr<<"usage: mpirun "<<argv[0]<<" <N> <K>"<<std::endl;
    return -1;
  }
  MPI_Init(&argc, &argv);
  // declare and init command line params
  long N, K;
  N = atol(argv[1]);
  K = atol(argv[2]);
 	double *sendRowBuff,*recvRowBuff,*sendColBuff,*recvColBuff,*recvRowBuff1,*recvColBuff1;
  int wrank,wsize;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
  MPI_Comm_size(MPI_COMM_WORLD, &wsize);

  long root_p = sqrt(wsize);        //wsize will always be perfect square number
  long rows = wrank / root_p;
  long columns = wrank % root_p;
  long block_size = N/root_p;

  MPI_Request* requests = new MPI_Request[8];
	MPI_Status* statuses = new MPI_Status[8];
	sendRowBuff=(double *)malloc(sizeof(double)*block_size);
	recvRowBuff=(double *)malloc(sizeof(double)*block_size);
	sendColBuff=(double *)malloc(sizeof(double)*block_size);
  recvColBuff=(double *)malloc(sizeof(double)*block_size);
  recvRowBuff1=(double *)malloc(sizeof(double)*block_size);
  recvColBuff1=(double *)malloc(sizeof(double)*block_size);

  // use double for heat 2d 
  bool res1 = true;
  double** H0 = new double*[block_size];
  
  for (long i=0; i<block_size; i++) {
    H0[i] = new double[block_size];
  }

  // write code here
  for (long row = 0; row<block_size; ++row) {
    for (long col=0; col<block_size; ++col) {
      H0[row][col] = genH0(row, col, N);
    }
  }
  double partial_result;
  double** H = new double*[block_size];
  for (long i=0; i<block_size; i++) {
    H[i] = new double[block_size];
  }
	long rowPrev, colPrev, rowNext, colNext;
  MPI_Barrier(MPI_COMM_WORLD);
  std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
  for (long it = 0; it<K; ++it) {
  	if(wsize==1){
  		for (long row = 0; row<block_size; row++) {
	  		for (long col=0; col<block_size; col++) {					  				
	  			rowPrev= (row==0) ? row : row-1; 
	  			rowNext= (row==block_size-1) ? row : row+1;
	  			colPrev= (col==0) ? col : col-1;
	  			colNext= (col==block_size-1) ? col : col+1;
	      	partial_result = (H0[row][col]+H0[row][colPrev]+H0[row][colNext]+H0[rowPrev][col]+H0[rowNext][col]);
	      	H[row][col] = partial_result/5;
	      //	cout<<H[row][col]<<" ";
	    	}
	    //	cout<<endl;
   		}
  	} else if(wrank==0){																												//upper leftmost
  		//send and receive only from 2 sides  		
  		MPI_Irecv(recvColBuff, block_size, MPI_DOUBLE, wrank+1, 0, MPI_COMM_WORLD, &requests[0]);
    	MPI_Irecv(recvRowBuff, block_size, MPI_DOUBLE, wrank+root_p, 0, MPI_COMM_WORLD, &requests[1]);

  		for(long i=0; i<block_size;++i) {
  			sendRowBuff[i]=H0[block_size- 1][i];
  			sendColBuff[i]=H0[i][block_size-1];
  		}
      MPI_Isend(sendColBuff, block_size, MPI_DOUBLE, wrank+1, 0, MPI_COMM_WORLD,&requests[2]);
      MPI_Isend(sendRowBuff, block_size, MPI_DOUBLE, wrank+root_p, 0, MPI_COMM_WORLD,&requests[3]);
      MPI_Waitall(2,requests,statuses);

    	for (long row = 0; row<block_size; row++) {
	  		for (long col=0; col<block_size; col++) {				
	  			rowPrev= (row==0) ? row : row-1; 
	  			rowNext= (row==block_size-1) ? recvRowBuff[col] : row+1;
	  			colPrev= (col==0) ? col : col-1;
	  			colNext= (col==block_size-1) ? recvColBuff[row] : col+1;

	      	partial_result = (H0[row][col]+H0[row][colPrev]+H0[row][colNext]+H0[rowPrev][col]+H0[rowNext][col]);
	      	H[row][col] = partial_result/5;
	    	}
   		}

  	} 
  	else if(wrank==root_p-1){																										//upper rightmost
  		//send and receive only from 2 sides
  		MPI_Irecv(recvColBuff, block_size, MPI_DOUBLE, wrank-1, 0, MPI_COMM_WORLD, &requests[0]);
  		MPI_Irecv(recvRowBuff, block_size, MPI_DOUBLE, wrank+root_p, 0, MPI_COMM_WORLD, &requests[1]);
  		for(long i=0; i<block_size;++i) {
  			sendRowBuff[i]=H0[block_size- 1][i];
  			sendColBuff[i]=H0[i][0];
  		}
  		MPI_Isend(sendRowBuff, block_size, MPI_DOUBLE, wrank+root_p, 0, MPI_COMM_WORLD,&requests[2]);
      MPI_Isend(recvColBuff, block_size, MPI_DOUBLE, wrank-1, 0, MPI_COMM_WORLD,&requests[3]);
      MPI_Waitall(2,requests,statuses);
			for (long row = 0; row<block_size; row++) {
	  		for (long col=0; col<block_size; col++) {				
	  			rowPrev= (row==0) ? row : row-1; 
	  			rowNext= (row==block_size-1) ? recvRowBuff[col] : row+1;
	  			colPrev= (col==0) ? recvColBuff[row] : col-1;
	  			colNext= (col==block_size-1) ? col : col+1;

	      	partial_result = (H0[row][col]+H0[row][colPrev]+H0[row][colNext]+H0[rowPrev][col]+H0[rowNext][col]);
	      	H[row][col] = partial_result/5;
	    	}
   		}
  	} else if(wrank==wsize-1){																	//lower rightmost
  		//send and receive only from 2 sides
  		MPI_Irecv(recvRowBuff, block_size, MPI_DOUBLE, wrank-root_p, 0, MPI_COMM_WORLD, &requests[0]);
      MPI_Irecv(recvColBuff, block_size, MPI_DOUBLE, wrank-1, 0, MPI_COMM_WORLD, &requests[1]);
  		for(long i=0; i<block_size;++i) {
  			sendRowBuff[i]=H0[0][i];
  			sendColBuff[i]=H0[i][0];
  		}
      MPI_Isend(sendRowBuff, block_size, MPI_DOUBLE, wrank-root_p, 0, MPI_COMM_WORLD,&requests[2]);
      MPI_Isend(sendColBuff, block_size, MPI_DOUBLE, wrank-1, 0, MPI_COMM_WORLD,&requests[3]);
      MPI_Waitall(2,requests,statuses);
      for (long row = 0; row<block_size; row++) {
	  		for (long col=0; col<block_size; col++) {
	  			rowPrev= (row==0) ? recvRowBuff[col] : row-1; 
	  			rowNext= (row==block_size-1) ? row : row+1;
	  			colPrev= (col==0) ? recvColBuff[row] : col-1;
	  			colNext= (col==block_size-1) ? col : col+1;

	      	partial_result = (H0[row][col]+H0[row][colPrev]+H0[row][colNext]+H0[rowPrev][col]+H0[rowNext][col]);
	      	H[row][col] = partial_result/5;
	    	}
   		}
  	} else if(wrank==wsize - root_p){														//lower leftmost
  		//send and receive only from 2 sides
  		MPI_Irecv(recvRowBuff, block_size, MPI_DOUBLE, wrank-root_p, 0, MPI_COMM_WORLD, &requests[0]);
      MPI_Irecv(recvColBuff, block_size, MPI_DOUBLE, wrank+1, 0, MPI_COMM_WORLD, &requests[1]);

  		for(long i=0; i<block_size;++i) {
  			sendRowBuff[i]=H0[0][i];
  			sendColBuff[i]=H0[i][block_size-1];
  		}
      MPI_Isend(sendRowBuff, block_size, MPI_DOUBLE, wrank-root_p, 0, MPI_COMM_WORLD,&requests[2]);
      MPI_Isend(sendColBuff, block_size, MPI_DOUBLE, wrank+1, 0, MPI_COMM_WORLD,&requests[3]);
      MPI_Waitall(2,requests,statuses);
      for (long row = 0; row<block_size; row++) {
	  		for (long col=0; col<block_size; col++) {
	  			rowPrev= (row==0) ? recvRowBuff[col] : row-1; 
	  			rowNext= (row==block_size-1) ? row : row+1;
	  			colPrev= (col==0) ? col : col-1;
	  			colNext= (col==block_size-1) ? recvColBuff[row] : col+1;

	      	partial_result = (H0[row][col]+H0[row][colPrev]+H0[row][colNext]+H0[rowPrev][col]+H0[rowNext][col]);
	      	H[row][col] = partial_result/5;
	    	}
   		} 
  	} else if((wrank%root_p)==0){																								//left middle side 
  		//send and receive only from 3 sides
  		MPI_Irecv(recvRowBuff, block_size, MPI_DOUBLE, wrank-root_p, 0, MPI_COMM_WORLD, &requests[0]);
      MPI_Irecv(recvColBuff, block_size, MPI_DOUBLE, wrank+1, 0, MPI_COMM_WORLD, &requests[1]);  		
  		MPI_Irecv(recvRowBuff1, block_size, MPI_DOUBLE, wrank+root_p, 0, MPI_COMM_WORLD, &requests[2]);

      for(long i=0; i<block_size;++i) {
  			sendRowBuff[i]=H0[0][i];
  			sendColBuff[i]=H0[i][block_size-1];
  		}
  		MPI_Isend(sendRowBuff, block_size, MPI_DOUBLE, wrank-root_p, 0, MPI_COMM_WORLD,&requests[3]);//upper side row
  		MPI_Isend(sendColBuff, block_size, MPI_DOUBLE, wrank+1, 0, MPI_COMM_WORLD,&requests[4]);
  		for(long i=0; i<block_size;++i) {
  			sendRowBuff[i]=H0[block_size-1][i];
  		}
  		MPI_Isend(sendRowBuff, block_size, MPI_DOUBLE, wrank+root_p, 0, MPI_COMM_WORLD,&requests[5]);//bottom side row
  		MPI_Waitall(3,requests,statuses);
  		for (long row = 0; row<block_size; row++) {
	  		for (long col=0; col<block_size; col++) {				
	  			rowPrev= (row==0) ? recvRowBuff[col] : row-1; 
	  			rowNext= (row==block_size-1) ? recvRowBuff1[col] : row+1;
	  			colPrev= (col==0) ? col : col-1;
	  			colNext= (col==block_size-1) ? recvColBuff[row] : col+1;	

	      	partial_result = (H0[row][col]+H0[row][colPrev]+H0[row][colNext]+H0[rowPrev][col]+H0[rowNext][col]);
	      	H[row][col] = partial_result/5;
	    	}
   		}
  	} else if((wrank+1)%root_p==0){																							//right middle side 
  		//send and receive only from 3 sides
  		MPI_Irecv(recvRowBuff, block_size, MPI_DOUBLE, wrank-root_p, 0, MPI_COMM_WORLD, &requests[0]);
      MPI_Irecv(recvColBuff, block_size, MPI_DOUBLE, wrank-1, 0, MPI_COMM_WORLD, &requests[1]);  		
  		MPI_Irecv(recvRowBuff1, block_size, MPI_DOUBLE, wrank+root_p, 0, MPI_COMM_WORLD, &requests[2]);

      for(long i=0; i<block_size;++i) {
  			sendRowBuff[i]=H0[0][i];
  			sendColBuff[i]=H0[i][0];
  		}
  		MPI_Isend(sendRowBuff, block_size, MPI_DOUBLE, wrank-root_p, 0, MPI_COMM_WORLD,&requests[3]);//upper side row
  		MPI_Isend(sendColBuff, block_size, MPI_DOUBLE, wrank-1, 0, MPI_COMM_WORLD,&requests[4]);
  		for(long i=0; i<block_size;++i) {
  			sendRowBuff[i]=H0[block_size-1][i];
  		}
  		MPI_Isend(sendRowBuff, block_size, MPI_DOUBLE, wrank+root_p, 0, MPI_COMM_WORLD,&requests[5]);//bottom side row
  		MPI_Waitall(3,requests,statuses);
  		for (long row = 0; row<block_size; row++) {
	  		for (long col=0; col<block_size; col++) {				
	  			rowPrev= (row==0) ? recvRowBuff[col] : row-1; 
	  			rowNext= (row==block_size-1) ? recvRowBuff1[col] : row+1;
	  			colPrev= (col==0) ? recvColBuff[row] : col-1;
	  			colNext= (col==block_size-1) ? col : col+1;	

	      	partial_result = (H0[row][col]+H0[row][colPrev]+H0[row][colNext]+H0[rowPrev][col]+H0[rowNext][col]);
	      	H[row][col] = partial_result/5;
	    	}
   		}
  	} else if(wrank>(wsize - root_p) && wrank<(wsize-1)){							//bottom middle side 
  		//send and receive only from 3 sides
      MPI_Irecv(recvRowBuff, block_size, MPI_DOUBLE, wrank-root_p, 0, MPI_COMM_WORLD, &requests[0]);  		
  		MPI_Irecv(recvColBuff, block_size, MPI_DOUBLE, wrank-1, 0, MPI_COMM_WORLD, &requests[1]);
  		MPI_Irecv(recvColBuff1, block_size, MPI_DOUBLE, wrank+1, 0, MPI_COMM_WORLD, &requests[2]);
      for(long i=0; i<block_size;++i) {
  			sendRowBuff[i]=H0[0][i];
  			sendColBuff[i]=H0[i][0];
  		}

      MPI_Isend(sendRowBuff, block_size, MPI_DOUBLE, wrank-root_p, 0, MPI_COMM_WORLD,&requests[3]);
  		MPI_Isend(sendColBuff, block_size, MPI_DOUBLE, wrank-1, 0, MPI_COMM_WORLD,&requests[4]);//left side column
  		for(long i=0; i<block_size;++i) {
  			sendColBuff[i]=H0[i][block_size-1];
  		}
      MPI_Isend(sendColBuff, block_size, MPI_DOUBLE, wrank+1, 0, MPI_COMM_WORLD,&requests[5]);//right side column
      MPI_Waitall(3,requests,statuses);
      for (long row = 0; row<block_size; row++) {
	  		for (long col=0; col<block_size; col++) {				
	  			rowPrev= (row==0) ? recvRowBuff[col] : row-1; 
	  			rowNext= (row==block_size-1) ? row : row+1;
	  			colPrev= (col==0) ? recvColBuff[row] : col-1;
	  			colNext= (col==block_size-1) ? recvColBuff1[row] : col+1;	

	      	partial_result = (H0[row][col]+H0[row][colPrev]+H0[row][colNext]+H0[rowPrev][col]+H0[rowNext][col]);
	      	H[row][col] = partial_result/5;
	    	}
   		}
  	} else if(wrank>0 && wrank<root_p-1){																					//upper middle side 
  		//send and receive only from 3 sides
  		MPI_Irecv(recvRowBuff, block_size, MPI_DOUBLE, wrank+root_p, 0, MPI_COMM_WORLD, &requests[0]);
  		MPI_Irecv(recvColBuff, block_size, MPI_DOUBLE, wrank-1, 0, MPI_COMM_WORLD, &requests[1]);
  		MPI_Irecv(recvColBuff1, block_size, MPI_DOUBLE, wrank+1, 0, MPI_COMM_WORLD, &requests[2]);

      for(long i=0; i<block_size;++i) {
  			sendRowBuff[i]=H0[block_size-1][i];
  			sendColBuff[i]=H0[i][0];
  		}

      MPI_Isend(sendRowBuff, block_size, MPI_DOUBLE, wrank+root_p, 0, MPI_COMM_WORLD,&requests[3]);
  		MPI_Isend(sendColBuff, block_size, MPI_DOUBLE, wrank-1, 0, MPI_COMM_WORLD,&requests[4]);//left side column
  		for(long i=0; i<block_size;++i) {
  			sendColBuff[i]=H0[i][block_size-1];
  		}
      MPI_Isend(sendColBuff, block_size, MPI_DOUBLE, wrank+1, 0, MPI_COMM_WORLD,&requests[5]);//right side column
      MPI_Waitall(3,requests,statuses);
      for (long row = 0; row<block_size; row++) {
	  		for (long col=0; col<block_size; col++) {				
	  			rowPrev= (row==0) ? row : row-1; 
	  			rowNext= (row==block_size-1) ? recvRowBuff[col] : row+1;
	  			colPrev= (col==0) ? recvColBuff[row] : col-1;
	  			colNext= (col==block_size-1) ? recvColBuff1[row] : col+1;	

	      	partial_result = (H0[row][col]+H0[row][colPrev]+H0[row][colNext]+H0[rowPrev][col]+H0[rowNext][col]);
	      	H[row][col] = partial_result/5;
	    	}
   		}
  	} else{
  		//send and receive from all sides
  		MPI_Irecv(recvRowBuff, block_size, MPI_DOUBLE, wrank-root_p, 0, MPI_COMM_WORLD, &requests[0]);
  		MPI_Irecv(recvColBuff, block_size, MPI_DOUBLE, wrank-1, 0, MPI_COMM_WORLD, &requests[1]);
  		MPI_Irecv(recvRowBuff1, block_size, MPI_DOUBLE, wrank+root_p, 0, MPI_COMM_WORLD, &requests[2]);
      MPI_Irecv(recvColBuff1, block_size, MPI_DOUBLE, wrank+1, 0, MPI_COMM_WORLD, &requests[3]);

      for(long i=0; i<block_size;++i) {
  			sendRowBuff[i]=H0[0][i];
  			sendColBuff[i]=H0[i][0];
  		}
  		MPI_Isend(sendRowBuff, block_size, MPI_DOUBLE, wrank-root_p, 0, MPI_COMM_WORLD,&requests[4]);//upper side row
  		MPI_Isend(sendColBuff, block_size, MPI_DOUBLE, wrank-1, 0, MPI_COMM_WORLD,&requests[5]);//left side column
  		for(long i=0; i<block_size;++i) {
  			sendRowBuff[i]=H0[block_size-1][i];
  			sendColBuff[i]=H0[i][block_size-1];
  		}
  		MPI_Isend(sendRowBuff, block_size, MPI_DOUBLE, wrank+root_p, 0, MPI_COMM_WORLD,&requests[6]);//bottom side row
  		MPI_Isend(sendColBuff, block_size, MPI_DOUBLE, wrank+1, 0, MPI_COMM_WORLD,&requests[7]);//right side column
  		MPI_Waitall(4,requests,statuses);
  		for (long row = 0; row<block_size; row++) {
	  		for (long col=0; col<block_size; col++) {				
	  			rowPrev= (row==0) ? recvRowBuff[col] : row-1; 
	  			rowNext= (row==block_size-1) ? recvRowBuff1[col] : row+1;
	  			colPrev= (col==0) ? recvColBuff[row] : col-1;
	  			colNext= (col==block_size-1) ? recvColBuff1[row] : col+1;	

	      	partial_result = (H0[row][col]+H0[row][colPrev]+H0[row][colNext]+H0[rowPrev][col]+H0[rowNext][col]);
	      	H[row][col] = partial_result/5;
	    	}
   		}
  	}
  	// res1= check2DHeat(H, N, 0, wsize, it);
   // 	if(res1)
   // 		break;
  	for (long row = 0; row<block_size; row++) {
	  		for (long col=0; col<block_size; col++) {	
	  			H0[row][col]=H[row][col];
	  		}
	  	} 	
	}
	if(wrank==0) {
    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cerr<<elapsed_seconds.count()<<std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
  
  delete[] H0;
  delete[] H;
  return 0;
}

