#include <ATen/ATen.h>

#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define EPSILON 0.000001
#define CROSS(dest,v1,v2) \
          dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
          dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
          dest[2]=v1[0]*v2[1]-v1[1]*v2[0];
#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])
#define SUB(dest,v1,v2) \
          dest[0]=v1[0]-v2[0]; \
          dest[1]=v1[1]-v2[1]; \
          dest[2]=v1[2]-v2[2];

namespace {

template<typename scalar_t>
static __inline__ __device__ scalar_t mag2(const scalar_t* x) {
    scalar_t l = 0;
    for (int i=0; i<3; ++i) {
        l += x[i] * x[i];
    }
    return l;
}

template<typename scalar_t>
static __inline__ __device__ scalar_t norm(const scalar_t* x) {
    scalar_t l = 0;
    for (int i=0; i<3; ++i) {
        l += x[i] * x[i];
    }
    return sqrt(l);
}

template<typename scalar_t>
static __inline__ __device__ scalar_t dist2(const scalar_t* x, const scalar_t* y) {
    scalar_t l = 0;
    scalar_t diff;
    for (int i=0; i<3; ++i) {
        diff = x[i] - y[i];
        l += diff * diff;
    }
    return l;
}

template<typename scalar_t>
static __inline__ __device__ scalar_t dist(const scalar_t* x, const scalar_t* y) {
    scalar_t l = 0;
    scalar_t diff;
    for (int i=0; i<3; ++i) {
        diff = x[i] - y[i];
        l += diff * diff;
    }
    return sqrt(l);
}


template<typename scalar_t>
static __inline__ __device__ scalar_t dot(const scalar_t* x, const scalar_t* y) {
    scalar_t l = 0;
    for (int i=0; i<3; ++i) {
        l += x[i] * y[i];
    }
    return l;
}


// find distance x0 is from segment x1-x2
template<typename scalar_t>
static __inline__ __device__ scalar_t point_segment_distance(const scalar_t* x0, const scalar_t* x1, const scalar_t* x2, scalar_t* r)
{
    scalar_t dx[3] = {x2[0]-x1[0], x2[1]-x1[1], x2[2]-x1[2]};
    scalar_t m2 = mag2(dx);
    // find parameter value of closest point on segment
    // scalar_t s12= (scalar_t) (dot(x2-x0, dx)/m2);
    scalar_t s12 = (scalar_t) (dot(x2, dx) - dot(x0, dx)) / m2;
    if (s12 < 0){
       s12 = 0;
    }
    else if (s12 > 1){
       s12 = 1;
    }
    for (int i=0; i < 3; ++i) {
        r[i] = s12*x1[i] + (1-s12) * x2[i];
    }
    // and find the distance
    return dist(x0, r);
}

/* the original jgt code */
template<typename scalar_t>
static __inline__ __device__ int intersect_triangle(
               const scalar_t* orig, const scalar_t* dir,
		       const scalar_t* vert0, const scalar_t* vert1,
               const scalar_t* vert2, scalar_t* t, scalar_t *u, scalar_t *v) {

    scalar_t edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
    scalar_t det,inv_det;
    
    /* find vectors for two edges sharing vert0 */
    SUB(edge1, vert1, vert0);
    SUB(edge2, vert2, vert0);
    
    /* begin calculating determinant - also used to calculate U parameter */
    CROSS(pvec, dir, edge2);
    
    /* if determinant is near zero, ray lies in plane of triangle */
    det = DOT(edge1, pvec);
    
    if (det > -EPSILON && det < EPSILON)
        return 0;
    inv_det = 1.0 / det;
    
    /* calculate distance from vert0 to ray origin */
    SUB(tvec, orig, vert0);
    
    /* calculate U parameter and test bounds */
    *u = DOT(tvec, pvec) * inv_det;
    if (*u < 0.0 || *u > 1.0)
        return 0;
    
    /* prepare to test V parameter */
    CROSS(qvec, tvec, edge1);
    
    /* calculate V parameter and test bounds */
    *v = DOT(dir, qvec) * inv_det;
    if (*v < 0.0 || (*u + *v) > 1.0)
        return 0;
    
    /* calculate t, ray intersects triangle */
    *t = DOT(edge2, qvec) * inv_det;
    
    return 1;
}

template<typename scalar_t>
static __inline__ __device__ int triangle_ray_intersection(const scalar_t* origin, const scalar_t* dest,
    const scalar_t* v1, const scalar_t* v2, const scalar_t* v3, scalar_t* t) {

    scalar_t _dir[3] = {dest[0] - origin[0], dest[1] - origin[1], dest[2] - origin[2]};

    // t is the distance, u and v are barycentric coordinates
    // http://fileadmin.cs.lth.se/cs/personal/tomas_akenine-moller/code/raytri_tam.pdf
    scalar_t u, v;
    return intersect_triangle(origin, _dir, v1, v2, v3, t, &u, &v);
}



// find distance x0 is from triangle x1-x2-x3
template<typename scalar_t>
// static scalar_t point_triangle_distance(const Vec3f &x0, const Vec3f &x1, const Vec3f &x2, const Vec3f &x3)
static __inline__ __device__ scalar_t point_triangle_distance(const scalar_t* x0, const scalar_t* x1, const scalar_t* x2, const scalar_t* x3, scalar_t* r) {
   // first find barycentric coordinates of closest point on infinite plane
    scalar_t x13[3];
    scalar_t x23[3];
    scalar_t x03[3];
    for (int i=0; i<3; ++i) {
        x13[i] = x1[i] - x3[i];
        x23[i] = x2[i] - x3[i];
        x03[i] = x0[i] - x3[i];
    }
    scalar_t m13 = mag2(x13);
    scalar_t m23 = mag2(x23);
    scalar_t m33 = mag2(x03);
    scalar_t d = dot(x13, x23);
    scalar_t invdet=1.f/max(m13*m23-d*d,1e-30f);
    scalar_t a = dot(x13, x03);
    scalar_t b = dot(x23, x03);
    // the barycentric coordinates themselves
    scalar_t w23=invdet*(m23*a-d*b);
    scalar_t w31=invdet*(m13*b-d*a);
    scalar_t w12=1-w23-w31;

    if (w23>=0 && w31>=0 && w12>=0){ // if we're inside the triangle
        for (int i=0; i<3; ++i) {
            r[i] = w23*x1[i] + w31*x2[i]+w12*x3[i];
        }
        return dist(x0, r); 
    }
    else { // we have to clamp to one of the edges
        scalar_t r1[3] = {0,0,0};
        scalar_t r2[3] = {0,0,0};
        if (w23 > 0) {// this rules out edge 2-3 for us
            scalar_t d1 = point_segment_distance(x0,x1,x2,r1);
            scalar_t d2 = point_segment_distance(x0,x1,x3,r2);
            if (d1 < d2) {
                for (int i=0; i < 3; ++i) {
                    r[i] = r1[i];
                }
                return d1;
            }
            else {
                for (int i=0; i < 3; ++i) {
                    r[i] = r2[i];
                }
                return d2;
            }
        }
        else if (w31 > 0) {// this rules out edge 1-3
            scalar_t d1 = point_segment_distance(x0,x1,x2,r1);
            scalar_t d2 = point_segment_distance(x0,x2,x3,r2);
            if (d1 < d2) {
                for (int i=0; i < 3; ++i) {
                    r[i] = r1[i];
                }
                return d1;
            }
            else {
                for (int i=0; i < 3; ++i) {
                    r[i] = r2[i];
                }
                return d2;
            }
        }
        else {// w12 must be >0, ruling out edge 1-2
            scalar_t d1 = point_segment_distance(x0,x1,x3,r1);
            scalar_t d2 = point_segment_distance(x0,x2,x3,r2);
            if (d1 < d2) {
                for (int i=0; i < 3; ++i) {
                    r[i] = r1[i];
                }
                return d1;
            }
            else {
                for (int i=0; i < 3; ++i) {
                    r[i] = r2[i];
                }
                return d2;
            }
        }
    }
}




// template<typename scalar_t>
// __global__ void sdf_cuda_kernel(
//         scalar_t* phi,  // 输出：存放每个体素格点的 SDF 值
//         const int32_t* faces, // 输入：三角面索引数组，长度 = num_faces * 3
//         const scalar_t* vertices, 
//         int batch_size,  // 批大小（有多少个独立网格要同时处理）
//         int num_faces,  // 三角面片数量
//         int num_vertices,  // 顶点数量
//         int grid_size) {  // 体素网格在每个维度上的分辨率 G

//     const int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= batch_size * grid_size * grid_size * grid_size) {
//         return;
//     }
//     const int i = tid % grid_size;
//     const int j = (tid / grid_size) % grid_size;
//     const int k = (tid / (grid_size*grid_size)) % grid_size;
//     const int bn = tid / (grid_size*grid_size*grid_size);
//     const scalar_t dx = 2./(grid_size-1);
//     const scalar_t center_x = -1 + (i + 0.5) * dx;
//     const scalar_t center_y = -1 + (j + 0.5) * dx;
//     const scalar_t center_z = -1 + (k + 0.5) * dx;

//     const scalar_t center[3] = {center_x, center_y, center_z};
//     int num_intersect = 0; // 用来统计射线与网格三角面的交点数
//     scalar_t min_distance=1000; // 记录离该体素中心最近的三角面距离，初始化一个很大值


//     // 遍历所有三角面，计算 SDF
//     for (int f = 0; f < num_faces; ++f) {
//         // 每个 face 占 3 个 int
//         const int32_t* face = &faces[3*f];  
//         // 拿到该面三个顶点的全局索引
//         const int v1i = face[0];
//         const int v2i = face[1];
//         const int v3i = face[2];
//         // 根据 bn(批)、顶点索引、3 维偏移，定位到 vertices 数组里的具体坐标
//         const scalar_t* v1 = &vertices[bn*num_vertices*3 + v1i*3];
//         const scalar_t* v2 = &vertices[bn*num_vertices*3 + v2i*3];
//         const scalar_t* v3 = &vertices[bn*num_vertices*3 + v3i*3];

//         // 1) 计算体素中心到三角形的最近点
//         scalar_t closest_point[3];
//         point_triangle_distance(center, v1, v2, v3, closest_point);
//         scalar_t distance = dist(center, closest_point);
//                 // 3) 更新全局最小距离
//         if (distance < min_distance) {
//             min_distance = distance;
//         }

//          // 奇偶交点法：从中心向网格外角发射射线，看与三角面相交几次
//         scalar_t origin[3] = {-1.0, -1.0, -1.0};
//         bool intersect = triangle_ray_intersection(center, origin, v1, v2, v3, &distance); 

//         if (intersect && distance >= 0) {
//             num_intersect++;
//         }
//     }
//     // 射线与网格相交次数偶数 => 网格外部，保留正距离；奇数 => 网格内部，把距离取负
//     if (num_intersect % 2 == 0) { 
//         min_distance = 0; //min_distance; //0.;
//     }
//     if (num_intersect % 2 == 1) {
//         min_distance *= -1;
//         // printf("min_distance: %f\n", min_distance);
//     }
//     // phi[tid] = (scalar_t) num_intersect;
//     // phi[bn*grid_size*grid_size*grid_size + k*grid_size*grid_size + j*grid_size + i] = min_distance;
//     phi[tid] = min_distance;  
//     // printf("min_distance: %f\n", min_distance);
//     // if (num_intersect % 2 == 0) {  
//     //     phi[tid] = 0;
//     // }
// } }


template<typename scalar_t>
__global__ void sdf_cuda_kernel(
        scalar_t* phi,  // 输出：存放每个体素格点的 SDF 值
        const int32_t* faces, // 输入：三角面索引数组，长度 = num_faces * 3
        const scalar_t* vertices, 
        const scalar_t* points,
        int batch_size,  // 批大小（有多少个独立网格要同时处理）
        int num_faces,  // 三角面片数量
        int num_vertices,  // 顶点数量
        int num_points) { 

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * num_points) {
        return;
    }
    int point_idx = tid % num_points;
    int bn = tid / num_points;

    if (bn >= batch_size || point_idx >= num_points) {
        printf("Illegal access: bn=%d/%d, point_idx=%d/%d, tid=%d\n", bn, batch_size, point_idx, num_points, tid);
        return;
    } //????????????????


    const scalar_t* point = &points[bn * num_points * 3 + point_idx * 3];



    int num_intersect = 0; // 用来统计射线与网格三角面的交点数
    scalar_t min_distance = 1000; // 记录离该体素中心最近的三角面距离，初始化一个很大值


    // 遍历所有三角面，计算 SDF
    for (int f = 0; f < num_faces; ++f) {
        // 每个 face 占 3 个 int
        const int32_t* face = &faces[3*f];  
        // 拿到该面三个顶点的全局索引
        const int v1i = face[0];
        const int v2i = face[1];
        const int v3i = face[2];
        if (v1i >= num_vertices || v2i >= num_vertices || v3i >= num_vertices) {
            printf("Face index out of range! face=(%d, %d, %d), num_vertices=%d\n", v1i, v2i, v3i, num_vertices);
            continue;
        }
        
        // 根据 bn(批)、顶点索引、3 维偏移，定位到 vertices 数组里的具体坐标
        const scalar_t* v1 = &vertices[bn*num_vertices*3 + v1i*3];
        const scalar_t* v2 = &vertices[bn*num_vertices*3 + v2i*3];
        const scalar_t* v3 = &vertices[bn*num_vertices*3 + v3i*3];

        // 1) 计算体素中心到三角形的最近点
        scalar_t closest_point[3];
        point_triangle_distance(point, v1, v2, v3, closest_point);
        scalar_t distance = dist(point, closest_point);
                // 3) 更新全局最小距离
        if (distance < min_distance) {
            min_distance = distance;
        }

         // 奇偶交点法：从中心向网格外角发射射线，看与三角面相交几次
        scalar_t origin[3] = {point[0], point[1], point[2]};
        scalar_t dest[3] = {-1.0, -1.0, -1.0};
        bool intersect = triangle_ray_intersection(origin, dest, v1, v2, v3, &distance);
        
        if (intersect && distance >= 0) {
            num_intersect++;
        }
    }
    if (num_intersect % 2 == 0) {  // 射线与网格相交次数偶数 => 网格外部，保留正距离；
        min_distance = 0; 
    }
    else { //奇数 => 网格内部，把距离取负
        min_distance *= -1;
    }
    phi[tid] = min_distance;  
} 
}


// at::Tensor sdf_cuda(
//         at::Tensor phi,
//         at::Tensor faces,
//         at::Tensor vertices) {

//     const auto batch_size = phi.size(0);
//     const auto grid_size = phi.size(1);
//     const auto num_faces = faces.size(0);
//     const auto num_vertices = vertices.size(1);
//     const int threads = 512;
//     // const dim3 blocks ((batch_size * grid_size * grid_size * grid_size) / threads);
//     const int total_threads = batch_size * grid_size * grid_size * grid_size;
//     const int blocks = (total_threads + threads - 1) / threads;


//     AT_DISPATCH_FLOATING_TYPES(phi.scalar_type(), "sdf_cuda", ([&] {
//       sdf_cuda_kernel<scalar_t><<<blocks, threads>>>(
//           phi.data<scalar_t>(),
//           faces.data<int32_t>(),
//           vertices.data<scalar_t>(),
//           batch_size,
//           num_faces,
//           num_vertices,
//           grid_size);
//       }));

//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess)
//             printf("Error in sdf: %s\n", cudaGetErrorString(err));
//     return phi;
// } 
at::Tensor sdf_cuda(
        at::Tensor phi,//[batch_size, num_points] 用于存放计算出来的每个点到网格的距离（SDF值）。
        at::Tensor faces,
        at::Tensor vertices,
        at::Tensor points) { //[batch_size, num_points, 3] 用于存放需要计算 SDF 的点坐标。

    const auto batch_size = phi.size(0); 
    const auto num_points = phi.size(1);
    const auto num_faces = faces.size(0);
    const auto num_vertices = vertices.size(1);
    const int threads = 512;
    // const dim3 blocks ((batch_size * grid_size * grid_size * grid_size) / threads);
    const int total_threads =  batch_size * num_points;
    const int blocks = (total_threads + threads - 1) / threads;


    AT_DISPATCH_FLOATING_TYPES(phi.scalar_type(), "sdf_cuda", ([&] {
      sdf_cuda_kernel<scalar_t><<<blocks, threads>>>(
          phi.data<scalar_t>(),
          faces.data<int32_t>(),
          vertices.data<scalar_t>(),
          points.data<scalar_t>(),
          batch_size,
          num_faces,
          num_vertices,
          num_points);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error in sdf: %s\n", cudaGetErrorString(err));
    return phi;
} 
