#include <mpi.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <syslog.h>
#include <sys/stat.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#define M_E 2.7182818284590452354
#endif

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define MAX_NUMBER_OF_ACTIONS 100
#define NUMBER_OF_ACTION_TYPES 4
#define ACTION_COLOR_INVERSION 0
#define ACTION_BLUE_AND_RED_SWITCH 1
#define ACTION_GAUSSIAN_BLUR 2
#define ACTION_VERTICAL_FLIP 3

#include <jpeglib.h>

struct __attribute__((__packed__)) Pixel
{
    unsigned char R;
    unsigned char G;
    unsigned char B;
};

struct ContextReadJPG
{
    char *filename;
    FILE *file;
    struct jpeg_error_mgr err;
    struct jpeg_decompress_struct info;

    unsigned long x;
    unsigned long y;
    int channels; //  3 =>RGB (for jpeg is only 3)   4 =>RGBA

    unsigned char *data;
    struct Pixel *pixelData;
};

struct ContextWriteJPG
{
    char *filename;
    FILE *file;
    struct jpeg_error_mgr err;
    struct jpeg_compress_struct info;
};

struct GaussianKernel
{
    unsigned char radius;
    double* matrix;
};

struct ModuleContext
{
    struct CommandLine
    {
        int argc;
        char **argv;

        char actions[MAX_NUMBER_OF_ACTIONS];
        unsigned int actionsNumber;
    } cmd;

    struct 
    {
        int size;
        int rank;
    } world;

    struct Processor
    {
        char name[MPI_MAX_PROCESSOR_NAME];
        int length;
    } processor;

    struct JPGContext
    {
        struct ContextReadJPG read;

        unsigned long long totalSize;
        unsigned long long partialSize;
        unsigned long long partialHeight;
        unsigned long long partialWidth;

        unsigned char *partialBuffer;
        unsigned int gaussianBlurRadius;
    } jpg;
};

boolean CleanReadContext(struct ContextReadJPG *context)
{
    jpeg_destroy_decompress(&context->info);
    fclose(context->file);
    free(context->data);
    return TRUE;
}

boolean CleanWriteContext(struct ContextWriteJPG *context)
{
    jpeg_destroy_compress(&context->info);
    fclose(context->file);
    return TRUE;
}

boolean ReadJPEG(struct ContextReadJPG *context)
{
    context->file = fopen(context->filename, "rb");
    if (!context->file)
    {
        printf("Error reading JPEG file %s!", context->filename);
        return FALSE;
    }

    context->info.err = jpeg_std_error(&context->err);
    jpeg_create_decompress(&context->info);

    jpeg_stdio_src(&context->info, context->file);
    jpeg_read_header(&context->info, TRUE); // read jpeg file header
    jpeg_start_decompress(&context->info);  // decompress the file

    context->x = context->info.output_width;
    context->y = context->info.output_height;
    context->channels = context->info.num_components; //  3 =>RGB (for jpeg is only 3)   4 =>RGBA

    unsigned long data_size = context->x * context->y * context->channels;
    context->data = (unsigned char *)malloc(data_size);
    unsigned char *rowptr[1]; // pointer to an array
    while (context->info.output_scanline < context->info.output_height)
    {
        rowptr[0] = (unsigned char *)context->data + context->channels * context->info.output_width * context->info.output_scanline;
        jpeg_read_scanlines(&context->info, rowptr, 1);
    }

    jpeg_finish_decompress(&context->info); // finish decompressing
    context->pixelData = (struct Pixel *)context->data;

    return TRUE;
}

void SwapPixels(struct Pixel *p1, struct Pixel *p2);

boolean WriteJPEG(struct ContextReadJPG *contextInput)
{
    struct ContextWriteJPG contextOutput;
    contextOutput.filename = malloc(strlen(contextInput->filename) + strlen(".out.jpg") + 1);
    memset(contextOutput.filename, 0, strlen(contextInput->filename) + strlen(".out.jpg") + 1);
    memcpy(contextOutput.filename, contextInput->filename, strlen(contextInput->filename));
    memcpy(contextOutput.filename + strlen(contextInput->filename), ".out.jpg", strlen(".out.jpg"));

    contextOutput.file = fopen(contextOutput.filename, "wb");
    if (contextOutput.file == NULL)
    {
        return FALSE;
    }

    contextOutput.info.err = jpeg_std_error(&contextOutput.err);
    jpeg_create_compress(&contextOutput.info);

    jpeg_stdio_dest(&contextOutput.info, contextOutput.file);

    contextOutput.info.image_width = contextInput->info.image_width;
    contextOutput.info.image_height = contextInput->info.image_height;
    contextOutput.info.input_components = contextInput->info.num_components;
    contextOutput.info.in_color_space = JCS_RGB;

    jpeg_set_defaults(&contextOutput.info);
    jpeg_set_quality(&contextOutput.info, 100, TRUE);

    jpeg_start_compress(&contextOutput.info, TRUE);

    unsigned char *lpRowBuffer[1];
    while (contextOutput.info.next_scanline < contextOutput.info.image_height)
    {
        lpRowBuffer[0] = 
            &(contextInput->data[
                contextOutput.info.next_scanline * 
                contextInput->info.image_width * 
                contextInput->channels
                ]
            );       
        jpeg_write_scanlines(&contextOutput.info, lpRowBuffer, 1);
    }

    CleanWriteContext(&contextOutput);

    return TRUE;
}

boolean ApplyColorInversionOnJPEG(struct ContextReadJPG *context)
{
    for (int x = 0; x < context->info.image_height; x++)
    {
        for (int y = 0; y < context->info.image_width; y++)
        {
            int matrixOffset = x * context->info.image_width + y;
            int matrixOffsetChannels = matrixOffset * context->info.num_components;

            struct Pixel *pixel = (struct Pixel *)(context->data + matrixOffsetChannels);
            pixel->R = 0xFF - pixel->R;
            pixel->G = 0xFF - pixel->G;
            pixel->B = 0xFF - pixel->B;
        }
    }
}

boolean ApplyColorInversionOnPartialBuffer(unsigned char *data, unsigned long long size, int channels)
{
    // printf("ApplyColorInversionOnPartialBuffer\n");
    for (size_t i = 0; i < size; i += channels)
    {
        struct Pixel *pixel = (struct Pixel *)(data + i);
        pixel->R = 0xFF - pixel->R;
        pixel->G = 0xFF - pixel->G;
        pixel->B = 0xFF - pixel->B;
    }
}

boolean ApplyBlueAndRedSwitchOnPartialBuffer(unsigned char *data, unsigned long long size, int channels)
{
    // printf("ApplyBlueAndRedSwitchOnPartialBuffer\n");
    for (size_t i = 0; i < size; i += channels)
    {
        struct Pixel *pixel = (struct Pixel *)(data + i);

        unsigned temp = pixel->B;
        pixel->B = pixel->R;
        pixel->R = temp;
    }

    return TRUE;
}

double GaussianModel(double x, double y, double sigma) {
    return 1. / exp(-(x * x + y * y) / (2 * sigma * sigma));
}

void CreateGaussianKernel(struct GaussianKernel *kernel, unsigned char radius)
{
    const int size = radius + radius + 1;
    kernel->radius = radius;
    kernel->matrix = calloc(size * size, sizeof(double));

    const double sigma = 1.0 * size / 2.57;
    const double s = 2.0 * sigma * sigma;
    double sum = 0;
    for (int i = 0; i < size; i++) 
    {
        for (int j = 0; j < size; j++) 
        {
            const double r = sqrt(i * i + j * j);
            kernel->matrix[i * size + j] = (exp(-(r * r)/ s)) / (M_PI * s);
            sum += kernel->matrix[i * size + j];
        }
    }

    // Normalize the kernel
    // This ensures that all of the values in the kernel together add up to 1
    for (int i = 0; i < size * size; i++)
    {
        kernel->matrix[i] /= sum;
    }

    // for (int x = 0; x < size; x++) 
    // {
    //     for (int y = 0; y < size; y++)  
    //     {
    //         printf("%f ", kernel->matrix[x * size + y]);
    //     }
    //     printf("\n");
    // }
}

void ApplyGaussianKernelOnBuffer(
    unsigned char *data, 
    unsigned long long size, 
    unsigned long long width, 
    unsigned long long height, 
    int channels, 
    struct GaussianKernel* kernel)
{
    // printf("Gaussian partial => size: %llu, width: %llu, height: %llu, channels: %d\n",
    //     size, width, height, channels);

    for (int i = 0; i < width; i++) 
    {
        for (int j = 0; j < height; j++) 
        {
            double b = 0;
            double g = 0;
            double r = 0;

            for (int m = -kernel->radius; m < kernel->radius; m++) 
            {
                for (int n = -kernel->radius; n < kernel->radius; n++) 
                {
                    const unsigned long long x = MIN(height - 1, MAX(0, m + i));
                    const unsigned long long y = MIN(width - 1, MAX(0, n + j));
                    const unsigned long long pixelOffset = (x * width + y) * channels;
                    struct Pixel *pixel = (struct Pixel *)(data + pixelOffset);

                    const double weight = 
                        kernel->matrix[(m + kernel->radius) * kernel->radius + (n + kernel->radius)];
                    
                    b += weight * pixel->B;
                    g += weight * pixel->G;
                    r += weight * pixel->R;
                }
            }

            struct Pixel *pixel = (struct Pixel *)(data + j * width * channels);
            pixel->B = ceil(b);
            pixel->G = ceil(g);
            pixel->R = ceil(r);
        }
    }
}

void SwapPixels(struct Pixel *p1, struct Pixel *p2)
{
    struct Pixel temp = *p1;
    *p1 = *p2;
    *p2 = temp;
}

void FlipMatrix(unsigned char *data, unsigned long long width, unsigned long long height, int channels)
{
    // vertical
    // printf("Flip matrix: width: %llu, height: %llu, channels: %d\n", width, height, channels);

    for (unsigned int row = 0; row < height; row++)
    {
        unsigned char *lpRowBuffer[1] = { &(data[row * (width * channels)]) };
        struct Pixel *pixels = (struct Pixel *)(lpRowBuffer[0]);

        for (unsigned int i = 0; i < width / 2; i++)
        {
            struct Pixel *pixel1 = pixels + i;
            struct Pixel *pixel2 = pixels + width - 1 - i;
            SwapPixels(pixel1, pixel2);
        }
    }
}

boolean ParseCommandLine(struct ModuleContext* ctx, int argc, char **argv)
{
    ctx->cmd.argc = argc;
    ctx->cmd.argv = argv;

    if (ctx->cmd.argc < 2)
    {
        printf("No file provided!\n");
        return FALSE;
    }

    if (ctx->cmd.argc < 3)
    {
        printf("No actions to execute on given file!\n");
        return FALSE;
    }

    ctx->cmd.actionsNumber = 0;
    for (unsigned int i = 0; i < argc; i++) // silently ignore invalid input
    {
        if (strlen(argv[i]) != 2)
        {
            continue;
        }

        if (argv[i][0] != 'a' || argv[i][1] < '0' || argv[i][1] > '9')
        {
            continue;
        }

        unsigned int num = argv[i][1] - '0';
        if (num < NUMBER_OF_ACTION_TYPES)
        {
            ctx->cmd.actions[ctx->cmd.actionsNumber++] = num; 
        }
    }

    if (ctx->cmd.actionsNumber == 0)
    {
        printf("No actions to execute on given file after arguments parsing!\n");
        return FALSE;
    }

    return TRUE;
}

void PrintMPIData(struct ModuleContext* ctx)
{
    printf("Process %s with rank %d out of %d processors\n", 
        ctx->processor.name, ctx->world.rank, ctx->world.size);
}

boolean ReadJPEGToContext(struct ModuleContext* ctx)
{
    ctx->jpg.read.filename = ctx->cmd.argv[1];
    if (ReadJPEG(&ctx->jpg.read) == FALSE)
    {
        return FALSE;
    }

    ctx->jpg.totalSize = 
        ctx->jpg.read.info.image_height * 
        ctx->jpg.read.info.image_width * 
        ctx->jpg.read.channels;

    ctx->jpg.partialSize   = ctx->jpg.totalSize / ctx->world.size;
    ctx->jpg.partialHeight = ctx->jpg.read.info.image_height / ctx->world.size;
    ctx->jpg.partialWidth  = ctx->jpg.read.info.image_width / ctx->world.size;

    printf("Initial matrix data: size: %llu, width: %u, height: %u channels: %d\n", 
        ctx->jpg.totalSize, 
        ctx->jpg.read.info.image_width, 
        ctx->jpg.read.info.image_height, 
        ctx->jpg.read.channels);

    return TRUE;
}

boolean SplitValidationJPEG(struct ModuleContext* ctx)
{
    // if ((ctx->jpg.totalSize % ctx->world.size) || 
    //     (ctx->jpg.read.info.image_height % ctx->world.size))
    // {
    //     printf("Cannot evenly divide the image between the processes!\n");
    //     return FALSE;
    // }

    return TRUE;
}

boolean APPLY_ACTION_COLOR_INVERSION(struct ModuleContext* ctx)
{
    MPI_Scatter(
        ctx->jpg.read.data, 
        ctx->jpg.partialSize, 
        MPI_UNSIGNED_CHAR, 
        ctx->jpg.partialBuffer, 
        ctx->jpg.partialSize, 
        MPI_UNSIGNED_CHAR, 
        0, 
        MPI_COMM_WORLD);

    boolean result =  
        ApplyColorInversionOnPartialBuffer(
            ctx->jpg.partialBuffer, 
            ctx->jpg.partialSize, 
            ctx->jpg.read.channels);

    MPI_Gather(
        ctx->jpg.partialBuffer, 
        ctx->jpg.partialSize, 
        MPI_UNSIGNED_CHAR, 
        ctx->jpg.read.data, 
        ctx->jpg.partialSize,
        MPI_UNSIGNED_CHAR, 
        0, 
        MPI_COMM_WORLD);

    return result;
}

boolean APPLY_ACTION_BLUE_AND_RED_SWITCH(struct ModuleContext* ctx)
{
    MPI_Scatter(
        ctx->jpg.read.data, 
        ctx->jpg.partialSize, 
        MPI_UNSIGNED_CHAR, 
        ctx->jpg.partialBuffer, 
        ctx->jpg.partialSize, 
        MPI_UNSIGNED_CHAR, 
        0, 
        MPI_COMM_WORLD);

    boolean result =  
        ApplyBlueAndRedSwitchOnPartialBuffer(
            ctx->jpg.partialBuffer, 
            ctx->jpg.partialSize, 
            ctx->jpg.read.channels);

    MPI_Gather(
        ctx->jpg.partialBuffer, 
        ctx->jpg.partialSize, 
        MPI_UNSIGNED_CHAR, 
        ctx->jpg.read.data, 
        ctx->jpg.partialSize,
        MPI_UNSIGNED_CHAR, 
        0, 
        MPI_COMM_WORLD);

    return result;
}

boolean APPLY_ACTION_GAUSSIAN_BLUR2(struct ModuleContext* ctx)
{
    MPI_Scatter(
           ctx->jpg.read.data, 
           ctx->jpg.partialSize, 
           MPI_UNSIGNED_CHAR, 
           ctx->jpg.partialBuffer, 
           ctx->jpg.partialSize, 
           MPI_UNSIGNED_CHAR, 
           0, 
           MPI_COMM_WORLD);

    struct GaussianKernel kernel;
    CreateGaussianKernel(&kernel, ctx->jpg.gaussianBlurRadius);

    ApplyGaussianKernelOnBuffer(
        ctx->jpg.partialBuffer, 
        ctx->jpg.partialSize, 
        ctx->jpg.partialWidth, 
        ctx->jpg.partialHeight, 
        ctx->jpg.read.channels, 
        &kernel);

    free(kernel.matrix);

    MPI_Gather(
        ctx->jpg.partialBuffer, 
        ctx->jpg.partialSize, 
        MPI_UNSIGNED_CHAR, 
        ctx->jpg.read.data, 
        ctx->jpg.partialSize,
        MPI_UNSIGNED_CHAR, 
        0, 
        MPI_COMM_WORLD);

    return TRUE;
}

boolean APPLY_ACTION_GAUSSIAN_BLUR(struct ModuleContext* ctx)
{
    const unsigned int rowsPerProcess =
        ctx->jpg.read.info.image_height / ctx->world.size;

    const unsigned long long gaussianChunkSize = 
        ctx->jpg.gaussianBlurRadius * 
            ctx->jpg.read.info.image_width * 
            ctx->jpg.read.channels * 
            (ctx->world.size > 1);

    // printf("APPLY_ACTION_GAUSSIAN_BLUR on rank: %d blur radius: %u\n", 
    //     ctx->world.rank, ctx->jpg.gaussianBlurRadius);

    struct GaussianKernel kernel;
    CreateGaussianKernel(&kernel, ctx->jpg.gaussianBlurRadius);

    if (ctx->world.rank == 0)
    {
        for (unsigned int p = 1; p < ctx->world.size; p++)
        {
            const unsigned long long offset = 
                p * ctx->jpg.partialSize - gaussianChunkSize;

            const unsigned char* data = ctx->jpg.read.data + offset;

            const unsigned long long gaussianPartialSize = 
                ctx->jpg.partialSize + 
                    gaussianChunkSize * (1 + (p != (ctx->world.size - 1)));

            // printf("APPLY_ACTION_GAUSSIAN_BLUR sending to rank: %d offset: %llu size: %llu\n", 
            //     p, offset, gaussianPartialSize);

            MPI_Send(
                data, 
                gaussianPartialSize,
                MPI_UNSIGNED_CHAR,
                p, 
                0, 
                MPI_COMM_WORLD);
        }

        ApplyGaussianKernelOnBuffer(
            ctx->jpg.read.data, 
            ctx->jpg.partialSize + gaussianChunkSize, 
            ctx->jpg.partialWidth, 
            ctx->jpg.partialHeight + 
                ctx->jpg.gaussianBlurRadius * 
                (ctx->world.size > 1), 
            ctx->jpg.read.channels, 
            &kernel);  
        
        for (unsigned int p = 1; p < ctx->world.size; p++)
        {
            // printf("APPLY_ACTION_GAUSSIAN_BLUR receiving from rank: %d\n", p);

            MPI_Recv(
                ctx->jpg.read.data + p * ctx->jpg.partialSize, 
                ctx->jpg.partialSize, 
                MPI_UNSIGNED_CHAR, 
                p, 
                0, 
                MPI_COMM_WORLD, 
                MPI_STATUS_IGNORE);
        }
    }
    else
    {
        const unsigned long long gaussianPartialSize = 
            ctx->jpg.partialSize + 
                gaussianChunkSize * (1 + (ctx->world.rank != (ctx->world.size - 1)));
                
        const unsigned long long gaussianPartialHeight =
            ctx->jpg.partialHeight + 
                ctx->jpg.gaussianBlurRadius * 
                (1 + (ctx->world.rank != (ctx->world.size - 1)));
        
        // printf("APPLY_ACTION_GAUSSIAN_BLUR receiving on rank: %d gPartialSize: %llu gPartialHeight: %llu partialSize: %llu, partialHeight: %llu\n", 
        //     ctx->world.rank, 
        //     gaussianPartialSize, 
        //     gaussianPartialHeight, 
        //     ctx->jpg.partialSize,
        //     ctx->jpg.partialHeight);

        MPI_Recv(
            ctx->jpg.partialBuffer, 
            gaussianPartialSize, 
            MPI_UNSIGNED_CHAR, 
            0, 
            0, 
            MPI_COMM_WORLD, 
            MPI_STATUS_IGNORE);

        ApplyGaussianKernelOnBuffer(
            ctx->jpg.partialBuffer, 
            gaussianPartialSize, 
            ctx->jpg.partialWidth, 
            gaussianPartialHeight, 
            ctx->jpg.read.channels, 
            &kernel);  

        // printf("APPLY_ACTION_GAUSSIAN_BLUR sending from rank: %d\n", 
        //     ctx->world.rank);

        MPI_Send(
            ctx->jpg.partialBuffer +
                gaussianChunkSize, 
            ctx->jpg.partialSize,
            MPI_UNSIGNED_CHAR,
            0, 
            0, 
            MPI_COMM_WORLD);
    }

    free(kernel.matrix);

    return TRUE;
}

boolean APPLY_ACTION_VERTICAL_FLIP(struct ModuleContext* ctx)
{
    /*
        - compute rows and chunk size for partial buffer (in bytes not pixels) to be
            sent to each process
        - action by ranks:
            - 0: 
                - send chunks
                - apply flip on own data chunk
                - receive data from other processes that applied the flip action
            - non 0:
                - receive chunk of data to process
                - flip data
                - send data back to rank 0 / master process
    */

    const unsigned int rowsPerProcess = 
        ctx->jpg.read.info.image_height / ctx->world.size;
    const unsigned long long chunkSize = 
        rowsPerProcess * ctx->jpg.read.info.image_width * ctx->jpg.read.channels;

    if (ctx->world.rank == 0)
    {
        struct Pixel* pixels = (struct Pixel*)ctx->jpg.read.data;

        for (unsigned int p = 1; p < ctx->world.size; p++)
        {
            struct Pixel* rows = 
                pixels + p * rowsPerProcess * ctx->jpg.read.info.image_width;

            MPI_Send(
                rows, 
                chunkSize,
                MPI_UNSIGNED_CHAR,
                p, 
                0, 
                MPI_COMM_WORLD);
        }

        FlipMatrix(
            ctx->jpg.read.data, 
            ctx->jpg.read.info.image_width, 
            rowsPerProcess, 
            ctx->jpg.read.channels);
        
        for (unsigned int p = 1; p < ctx->world.size; p++)
        {
            MPI_Recv(
                ctx->jpg.read.data + p * chunkSize, 
                chunkSize, 
                MPI_UNSIGNED_CHAR, 
                p, 
                0, 
                MPI_COMM_WORLD, 
                MPI_STATUS_IGNORE);
        }
    }
    else
    {
        MPI_Recv(
            ctx->jpg.partialBuffer, 
            chunkSize, 
            MPI_UNSIGNED_CHAR, 
            0, 
            0, 
            MPI_COMM_WORLD, 
            MPI_STATUS_IGNORE);

        FlipMatrix(
            ctx->jpg.partialBuffer, 
            ctx->jpg.read.info.image_width, 
            rowsPerProcess, 
            ctx->jpg.read.channels);

        MPI_Send(
            ctx->jpg.partialBuffer, 
            chunkSize,
            MPI_UNSIGNED_CHAR,
            0, 
            0, 
            MPI_COMM_WORLD);
    }

    return TRUE;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    struct ModuleContext mContext;
    MPI_Comm_size(MPI_COMM_WORLD, &mContext.world.size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mContext.world.rank);
    MPI_Get_processor_name(mContext.processor.name, &mContext.processor.length);
    PrintMPIData(&mContext);
    MPI_Barrier(MPI_COMM_WORLD);

    if (ParseCommandLine(&mContext, argc, argv) == FALSE)
    {
        MPI_Finalize();
        return 1;
    }

    if (mContext.world.rank == 0)
    {
        if (ReadJPEGToContext(&mContext) == FALSE)
        {
            MPI_Finalize();
            return 1;
        }
    
        if (SplitValidationJPEG(&mContext) == FALSE)
        {
            CleanReadContext(&mContext.jpg.read);
            MPI_Finalize();
            return 1;
        }
    }

    mContext.jpg.gaussianBlurRadius = 5;
    MPI_Bcast(&mContext, sizeof(mContext), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &mContext.world.rank);
    mContext.jpg.partialBuffer = 
        (unsigned char *)malloc(
            mContext.jpg.partialSize + 
            (mContext.jpg.gaussianBlurRadius * 
            mContext.jpg.read.info.image_width * 
            mContext.jpg.read.channels * 
            (mContext.world.size > 1) *
            (1 + (mContext.world.rank != (mContext.world.size - 1)))));
    
    double sums_of_times[100] = {0};

    int action = -1;

    for (unsigned int i = 0; i < mContext.cmd.actionsNumber; i++)
    {
        clock_t now;
        double time_taken = 0.0;
        action = mContext.cmd.actions[i]; // this is not always correct
        switch (mContext.cmd.actions[i])
        {
        case ACTION_COLOR_INVERSION:
            now = clock();
            APPLY_ACTION_COLOR_INVERSION(&mContext);
            time_taken = ((double)clock() - now) / CLOCKS_PER_SEC;
            break;
        case ACTION_BLUE_AND_RED_SWITCH:
            now = clock();
            APPLY_ACTION_BLUE_AND_RED_SWITCH(&mContext);
            time_taken = ((double)clock() - now) / CLOCKS_PER_SEC;
            break;
        case ACTION_GAUSSIAN_BLUR:
            now = clock();
            APPLY_ACTION_GAUSSIAN_BLUR(&mContext);
            time_taken = ((double)clock() - now) / CLOCKS_PER_SEC;
            break;
        case ACTION_VERTICAL_FLIP:
            now = clock();
            APPLY_ACTION_VERTICAL_FLIP(&mContext);
            time_taken = ((double)clock() - now) / CLOCKS_PER_SEC;
            break;        
        default:
            break;
        }
        printf("Action %u: %f (s) of iteration %d on rank: %d\n", mContext.cmd.actions[i], time_taken, i, mContext.world.rank);
        sums_of_times[mContext.world.rank] += time_taken;
    }

    if (mContext.world.rank == 0)
    {
        for (unsigned int p = 1; p < mContext.world.size; p++)
        {
            MPI_Recv(
                &sums_of_times[p], 
                1, 
                MPI_LONG_DOUBLE_INT, 
                p, 
                0, 
                MPI_COMM_WORLD, 
                MPI_STATUS_IGNORE);
        }
    }
    else
    {
        MPI_Send(
                &sums_of_times[mContext.world.rank], 
                1,
                MPI_LONG_DOUBLE_INT,
                0, 
                0, 
                MPI_COMM_WORLD);
    }

    
    if (mContext.world.rank == 0)
    {
        double sum_of_times = 0.0;
        for (unsigned int p = 0; p < mContext.world.size; p++)
        {
             sum_of_times += sums_of_times[p];
        }

        const double total_time_taken = sum_of_times / mContext.cmd.actionsNumber / mContext.world.size;
        printf("Action %u: arithmetic %f (s) of world size: %d\n", action, total_time_taken, mContext.world.size);

        WriteJPEG(&mContext.jpg.read);
        CleanReadContext(&mContext.jpg.read);
    }

    free(mContext.jpg.partialBuffer);

    MPI_Finalize();

    return 0;
}
