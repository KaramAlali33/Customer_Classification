#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>
#include <stdlib.h>
#include <time.h>

int main() {

    srand(time(NULL));

    int arr[150]; 
    for (int i = 0; i < 150; i++) {
        arr[i] = rand() % 500 + 1; 
    }
    int state = fork();

    if (state < 0) {
        printf("fork failed\n");
        return 1;
    }
    else if (state == 0) { 

        int min = 500;
        for (int i = 0; i < 150; i++) { 
            if (arr[i] % 2 == 0) {
                if (arr[i] < min) {
                    min = arr[i];
                }
            }
        }
        printf("\nHello, I'm the Child Process!\n");
        printf("Child : PID = %d, Parent PID = %d\n", getpid(), getppid());
        printf("Minimum even number in array: %d\n", min);
    }
    else { 
        int sum = 0;
        for (int i = 0; i < 150; i++) {
            if (arr[i] % 3 == 0) {
                sum += arr[i];
            }
        }
        printf("Hello, I'm the Parent Process!\n");
        printf("Parent process, pid = %u\n", getpid());
        printf("Child process, pid = %u\n", state);
        printf("Sum of multiples of 3 in array: %d\n", sum);

        wait(NULL); 
    }

    return 0;
}
