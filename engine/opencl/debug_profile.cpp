/* Quick debug tool: dump a GPU-built profile for a single cell and compare
 * with a CPU-built profile. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

static const char *cl_err(cl_int e) {
    switch(e){case CL_SUCCESS:return "OK";case CL_BUILD_PROGRAM_FAILURE:return "BUILD_FAIL";default:return "?";}}

#define CL_CHECK(x) do{cl_int _e=(x);if(_e!=CL_SUCCESS){fprintf(stderr,"CL err %s @%d\n",cl_err(_e),__LINE__);exit(1);}}while(0)

static inline int clamp_i(int v,int lo,int hi){return v<lo?lo:(v>hi?hi:v);}

int main(int argc, char**argv){
    if(argc<8){fprintf(stderr,"usage: %s surface width height tx_col tx_row rx_col rx_row [kernel_path]\n",argv[0]);return 1;}
    const char*surf_path=argv[1];
    int W=atoi(argv[2]),H=atoi(argv[3]);
    int tx_col=atoi(argv[4]),tx_row=atoi(argv[5]);
    int rx_col=atoi(argv[6]),rx_row=atoi(argv[7]);
    const char*kpath=argc>8?argv[8]:"engine/opencl/itm_kernel.cl";
    double resolution=2.5;

    // load surface
    std::vector<int16_t> surf((size_t)W*H);
    FILE*fp=fopen(surf_path,"rb");if(!fp){perror("fopen");return 1;}
    fread(surf.data(),2,surf.size(),fp);fclose(fp);

    // CPU profile
    int dx=rx_col-tx_col, dy=rx_row-tx_row;
    double dist_cells=sqrt((double)dx*dx+(double)dy*dy);
    double dist=std::max(resolution,dist_cells*resolution);
    int segments=std::max(1,(int)ceil(dist_cells));
    int stride=segments+16;
    std::vector<double> cpu_prof(stride);
    cpu_prof[0]=(double)segments;
    cpu_prof[1]=dist/segments;
    for(int i=0;i<=segments;i++){
        double t=(double)i/(double)segments;
        int c=clamp_i((int)llround((double)tx_col+(double)dx*t),0,W-1);
        int r=clamp_i((int)llround((double)tx_row+(double)dy*t),0,H-1);
        cpu_prof[i+2]=(double)surf[(size_t)r*W+c];
    }
    for(int i=segments+3;i<stride;i++)cpu_prof[i]=cpu_prof[segments+2];

    // OpenCL setup
    cl_uint np=0;clGetPlatformIDs(0,NULL,&np);
    std::vector<cl_platform_id> plats(np);clGetPlatformIDs(np,plats.data(),NULL);
    cl_platform_id plat=plats[0];
    cl_uint nd=0;clGetDeviceIDs(plat,CL_DEVICE_TYPE_ALL,0,NULL,&nd);
    std::vector<cl_device_id> devs(nd);clGetDeviceIDs(plat,CL_DEVICE_TYPE_ALL,nd,devs.data(),NULL);
    cl_device_id dev=devs[0];
    cl_int e;
    cl_context_properties props[]={CL_CONTEXT_PLATFORM,(cl_context_properties)plat,0};
    cl_context ctx=clCreateContext(props,1,&dev,NULL,NULL,&e);CL_CHECK(e);
    cl_command_queue q=clCreateCommandQueue(ctx,dev,0,&e);CL_CHECK(e);

    FILE*kf=fopen(kpath,"rb");if(!kf){perror("fopen kernel");return 1;}
    fseek(kf,0,SEEK_END);long ks=ftell(kf);fseek(kf,0,SEEK_SET);
    std::vector<char> src(ks+1);fread(src.data(),1,ks,kf);src[ks]=0;fclose(kf);
    const char*s[1]={src.data()};size_t l[1]={(size_t)ks};
    cl_program prog=clCreateProgramWithSource(ctx,1,s,l,&e);CL_CHECK(e);
    e=clBuildProgram(prog,1,&dev,"-DUSE_DOUBLE",NULL,NULL);
    if(e!=CL_SUCCESS){size_t ll=0;clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,0,NULL,&ll);
        std::vector<char>log(ll+1);clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,ll,log.data(),NULL);
        log[ll]=0;fprintf(stderr,"build: %s\n",log.data());return 1;}
    cl_kernel kern=clCreateKernel(prog,"debug_profile_kernel",&e);CL_CHECK(e);

    cl_mem d_surf=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,surf.size()*2,surf.data(),&e);CL_CHECK(e);
    cl_mem d_out=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,stride*8,NULL,&e);CL_CHECK(e);
    double res=resolution;
    CL_CHECK(clSetKernelArg(kern,0,sizeof(d_surf),&d_surf));
    CL_CHECK(clSetKernelArg(kern,1,sizeof(int),&W));
    CL_CHECK(clSetKernelArg(kern,2,sizeof(int),&H));
    CL_CHECK(clSetKernelArg(kern,3,sizeof(int),&tx_col));
    CL_CHECK(clSetKernelArg(kern,4,sizeof(int),&tx_row));
    CL_CHECK(clSetKernelArg(kern,5,sizeof(double),&res));
    CL_CHECK(clSetKernelArg(kern,6,sizeof(int),&rx_col));
    CL_CHECK(clSetKernelArg(kern,7,sizeof(int),&rx_row));
    CL_CHECK(clSetKernelArg(kern,8,sizeof(d_out),&d_out));
    CL_CHECK(clSetKernelArg(kern,9,sizeof(int),&stride));
    size_t gl=1;
    CL_CHECK(clEnqueueNDRangeKernel(q,kern,1,NULL,&gl,NULL,0,NULL,NULL));
    std::vector<double> gpu_prof(stride);
    CL_CHECK(clEnqueueReadBuffer(q,d_out,CL_TRUE,0,stride*8,gpu_prof.data(),0,NULL,NULL));

    // compare
    int diffs=0;
    printf("segments: cpu=%d gpu=%d (elev[0]: cpu=%.1f gpu=%.1f)\n",segments,(int)gpu_prof[0],cpu_prof[0],gpu_prof[0]);
    printf("elev[1]: cpu=%.17g gpu=%.17g\n",cpu_prof[1],gpu_prof[1]);
    for(int i=0;i<stride;i++){
        if(cpu_prof[i]!=gpu_prof[i]){
            diffs++;
            if(diffs<=20)printf("  [%d] cpu=%.6f gpu=%.6f\n",i,cpu_prof[i],gpu_prof[i]);
        }
    }
    printf("total diffs: %d / %d\n",diffs,stride);
    return 0;
}
