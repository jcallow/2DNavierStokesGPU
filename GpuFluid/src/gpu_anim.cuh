/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */


#ifndef __GPU_ANIM_H__
#define __GPU_ANIM_H__

#include "gl_helper.h"
#include "cuda.h"
#include "cuda_gl_interop.h"
#include <iostream>
#include "FluidData.h"


using namespace std;

__global__
void render(GridData* density, uchar4* bitmap) {

	  int index = threadIdx.x + blockIdx.x*blockDim.x;

	  int blue = (int)((1.0 - density->src[index])*255.0);
	  blue = max(min(blue, 255), 0);
	  bitmap[index].x = blue;
	  bitmap[index].y = blue;
	  bitmap[index].z = blue;
	  bitmap[index].w = 255;
}



PFNGLBINDBUFFERARBPROC    glBindBuffer     = NULL;
PFNGLDELETEBUFFERSARBPROC glDeleteBuffers  = NULL;
PFNGLGENBUFFERSARBPROC    glGenBuffers     = NULL;
PFNGLBUFFERDATAARBPROC    glBufferData     = NULL;

struct GPUAnimBitmap {
    GLuint  bufferObj;
    cudaGraphicsResource *resource;
    FluidData fluidData;
    void (*fAnim)(uchar4*, FluidData&);
    void (*animExit)();
    void (*clickDrag)(int,int,int,int);
    int     dragStartX, dragStartY;

    GPUAnimBitmap(FluidData &fluidData): fluidData(fluidData) {
      clickDrag = NULL;

      // first, find a CUDA device and set it to graphic interop
      cudaDeviceProp  prop;
      int dev;
      memset( &prop, 0, sizeof( cudaDeviceProp ) );
      prop.major = 1;
      prop.minor = 0;
      cudaChooseDevice( &dev, &prop );
      cudaGLSetGLDevice( dev );

      // a bug in the Windows GLUT implementation prevents us from
      // passing zero arguments to glutInit()
      int c=1;
      char* dummy = "name";

      glutInit( &c, &dummy );
      glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
      glutInitWindowSize( fluidData.width, fluidData.height );
      glutCreateWindow( "bitmap" );

      glBindBuffer    = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer");
      glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");
      glGenBuffers    = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers");
      glBufferData    = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData");


      glGenBuffers( 1, &bufferObj );
      glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj );
      glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, fluidData.width * fluidData.height * 4,
                    NULL, GL_DYNAMIC_DRAW_ARB );

      cudaGraphicsGLRegisterBuffer( &resource, bufferObj, cudaGraphicsMapFlagsNone );
    }

    ~GPUAnimBitmap() {
    //    free_resources();
    }

    void free_resources( void ) {
        cudaGraphicsUnregisterResource( resource );

        glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
        glDeleteBuffers( 1, &bufferObj );
    }


    long image_size( void ) const { return fluidData.width * fluidData.height * 4; }

    void click_drag( void (*f)(int,int,int,int)) {
        clickDrag = f;
    }

    void anim_and_exit( void (*f)(uchar4*, FluidData &fluidData), void(*e)() ) {
        GPUAnimBitmap**   bitmap = get_bitmap_ptr();
        *bitmap = this;
        fAnim = f;
        animExit = e;

        glutKeyboardFunc( Key );
        glutDisplayFunc( Draw );
        if (clickDrag != NULL)
            glutMouseFunc( mouse_func );
        glutIdleFunc( idle_func );
        glutMainLoop();
    }

    // static method used for glut callbacks
    static GPUAnimBitmap** get_bitmap_ptr( void ) {
        static GPUAnimBitmap*   gBitmap;
        return &gBitmap;
    }

    // static method used for glut callbacks
    static void mouse_func( int button, int state,
                            int mx, int my ) {
        if (button == GLUT_LEFT_BUTTON) {
            GPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
            if (state == GLUT_DOWN) {
                bitmap->dragStartX = mx;
                bitmap->dragStartY = my;
            } else if (state == GLUT_UP) {
                bitmap->clickDrag( bitmap->dragStartX,
                                   bitmap->dragStartY,
                                   mx, my );
            }
        }
    }

    // static method used for glut callbacks
    static void idle_func( void ) {
        GPUAnimBitmap*  bitmap = *(get_bitmap_ptr());
        uchar4*         devPtr;
        size_t  size;

        cudaGraphicsMapResources( 1, &(bitmap->resource), NULL );
        cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, &size, bitmap->resource);
        bitmap->fAnim(devPtr, bitmap->fluidData);

        cudaGraphicsUnmapResources( 1, &(bitmap->resource), NULL );

        glutPostRedisplay();
    }

    // static method used for glut callbacks
    static void Key(unsigned char key, int x, int y) {
        switch (key) {
            case 27:
                GPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
                if (bitmap->animExit)
                    bitmap->animExit();
                bitmap->free_resources();
                exit(0);
        }
    }

    // static method used for glut callbacks
    static void Draw( void ) {
        GPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
        glClearColor( 0.0, 0.0, 0.0, 1.0 );
        glClear( GL_COLOR_BUFFER_BIT );
        glDrawPixels( bitmap->fluidData.width, bitmap->fluidData.height, GL_RGBA,
                      GL_UNSIGNED_BYTE, 0 );
        glutSwapBuffers();
    }

};


#endif  // __GPU_ANIM_H__

