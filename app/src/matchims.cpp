/*
This is the Driver program that computes the similarity between the target image and other images by accessing the vector/feature database.
*/

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>

int main(int argc, char *argv[]) {
    
    char dirname[256];
    char targetImg[256];
    char similarityMethod[256];
    char buffer[256];
    
    // check for sufficient arguments
    if( argc < 2) {
        printf("usage: %s <Target image path>\n", argv[0]);
        exit(-1);
    }
    // get the arguements path
    strcpy(targetImg, argv[2]);

    // open the directory
    
    /*
    iterate over the feture database CSV file and compute similarity between target image and every other image.
    */

    printf("Terminating\n");
    return(0);
}
