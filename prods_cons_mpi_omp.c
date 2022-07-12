/*
 *  main() should do the following:
 *
 *  1. starts N producers as per the first argument (default 1)
 *  2. starts N consumers as per the second argument (default 1)
 *
 *  The producer reads positive integers from standard input and passes those
 *  into the buffer.  The consumers read those integers and "perform a
 *  command" based on them.
 *
 *  on EOF of stdin, the first producer passes N copies of -1 into the buffer.
 *  The consumers interpret -1 as a signal to exit.
 */

#include <stdio.h>
#include <stdlib.h>             /* atoi() */
#include <unistd.h>             /* usleep() */
#include <assert.h>             /* assert() */
#include <signal.h>             /* signal() */
#include <alloca.h>             /* alloca() */
#include <omp.h>                /* For OpenMP */
#include <mpi.h>                /* For MPI */


#define MAX_BUF_SIZE        10
#define MAX_NUM_PROCS       5
#define MAX_SEND_SIZE       20
int buffer[MAX_BUF_SIZE] = {0};
int location = 0;
int num_procs = -1, myid = -1;
char hostname[MPI_MAX_PROCESSOR_NAME];

void insert_data(int producerno, int number)
{
    int done = 0;
    while (!done) {
        #pragma omp critical
        {
            /* Wait until consumers consumed something from the buffer and there is space */
            if (location < 10 && (number > 0 || location == 0)) {
                buffer[location] = number;
                printf("Process: %d on host %s producer %d inserting %d at %d\n", myid, hostname, producerno, number, location);
                location++; done = 1;
            }
        }
    }
}

int extract_data(int consumerno)
{
    int done = 0;
    int value = -1;
    while (!done) {
        #pragma omp critical
        {
            /* Wait until producers have put something in the buffer */
            if (location > 0) {
                location--; value = buffer[location]; buffer[location] = 0; done = 1;
	            printf("Process: %d on host %s consumer %d extracting %d from %d\n", myid, hostname, consumerno, value, location);
            }
        }
    }
    return value;
}


/*
Each consumer reads and "interprets" numbers from the bounded buffer:
 - positive integer N: sleep for N * 100000ms
 - negative integer: exit
*/
void consumer(int nproducers, int nconsumers)
{
    int number = -1;
    int consumerno = -1;
    #pragma omp parallel num_threads(nconsumers) private(number, consumerno)
    {
        consumerno = omp_get_thread_num();
        while (1)
        {
            number = extract_data(consumerno);
            if (number < 0)
                break;

            //usleep(10 * number);  /* "interpret" command for development */
            usleep(100000 * number);  /* "interpret" command for submission */
            fflush(stdout);
        }
    }
    return;
}


/*
Each producer reads numbers from stdin and inserts them into the bounded
buffer. On EOF from stdin, a -1 is inserted for each consumer to tell them
to exit.
*/

#define MAXLINELEN 128

void distribute_items(int producerno, int * items, int count, int dest) {
    int i;
    if (dest == myid) {
        if (items[0] != -1) {
            for (i = 0; i < count; i++) insert_data(producerno, items[i]);
        }
    } else {
        MPI_Send(items, count, MPI_INT, dest, 1, MPI_COMM_WORLD);
    }
}

void master_mpi(int producerno, int nconsumers) {
    char tmp_buffer[MAXLINELEN];
    int send_buffer[MAX_SEND_SIZE];
    int i, number, send_sum, send_size = 0, dest = 1;
    int * balancer;
    balancer = (int*) malloc(num_procs * sizeof(int));
    for (i = num_procs-1; i >= 0; i--) { balancer[i] = 0; }

    while (fgets(tmp_buffer, MAXLINELEN, stdin) != NULL) {
        number = atoi(tmp_buffer);
        send_buffer[send_size] = number;
        send_size++;
        send_sum += number;
        if (send_size == MAX_SEND_SIZE) {
            // find the process with the smallest sum
            for (i = num_procs-1; i >= 0; i--) { if (balancer[i] < balancer[dest]) dest = i; }
            distribute_items(producerno, send_buffer, send_size, dest);
            balancer[dest] += send_sum;
            send_size = 0;
            send_sum = 0;
        }
    }
    if (send_size > 0) {  // reached EOF before MAX_SEND_SIZE
        // find the process with the smallest sum
        for (i = num_procs-1; i >= 0; i--) { if (balancer[i] < balancer[dest]) dest = i; }
        distribute_items(producerno, send_buffer, send_size, dest);
    }
    number = -1;
    for (dest = 0; dest < num_procs; dest++) distribute_items(producerno, &number, 1, dest);
    free(balancer);
}

void producer(int nproducers, int nconsumers)
{
    int rc, i, count, producerno;
    MPI_Status stat;
    producerno = omp_get_thread_num();
    int array[MAX_SEND_SIZE] = {0};

    if (myid == 0) {
        master_mpi(producerno, nconsumers);
    } else {
        while (1) {
            rc = MPI_Recv(array, MAX_SEND_SIZE, MPI_INT, 0, 1, MPI_COMM_WORLD, &stat);
            if (array[0] == -1) break;
            rc = MPI_Get_count(&stat, MPI_INT, &count);
            for (i = 0; i < count; i++) {
                insert_data(producerno, array[i]);
            }
        }
    }
    for (i = 0; i < nconsumers; i++) {
        insert_data(producerno, -1);
    }
}


int main(int argc, char *argv[])
{
    int tid = -1, len = 0;
    int nproducers = 1;
    int nconsumers = 1;

    if (argc != 3) {
        nconsumers = 27;
    } else {
        nconsumers = atoi(argv[2]);
        if (nproducers <= 0 || nconsumers <= 0) {
            fprintf(stderr, "Error: nproducers & nconsumers should be >= 1\n");
            exit (1);
        }
    }

    // MPI Initializations
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Get_processor_name(hostname, &len);
    if (num_procs > MAX_NUM_PROCS) {
        fprintf(stderr, "Error: Max num procs should <= 5\n");
        exit (1);
    }

    printf("main: nproducers = %d, nconsumers = %d\n", nproducers, nconsumers);
    fflush(stdout);
    omp_set_dynamic(0); omp_set_nested(1);
    #pragma omp parallel num_threads(2) private(tid)
    {
        tid = omp_get_thread_num();
        /* Spawn N Consumer OpenMP Threads */
        if (tid == 0) consumer(nproducers, nconsumers);
        /* Spawn N Producer OpenMP Threads */
        if (tid == 1) producer(nproducers, nconsumers);
    }

    /* Finalize and cleanup */
    MPI_Finalize();
    return(0);
}
