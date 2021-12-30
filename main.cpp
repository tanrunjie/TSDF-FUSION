#include <iostream>
#include <chrono>
#include "utils.hpp"

using namespace chrono;

//  integrate a TSDF voxel volumn with given depth images
void Integrate(float *cam_K, float *cam2base, unsigned short *depth_im,
               int im_height, int im_width,
               int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
               float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z,
               float voxel_size, float trunc_margin, float *voxel_grid_TSDF, float *voxel_grid_weight)
{

    for (int pt_grid_x = 0; pt_grid_x < voxel_grid_dim_x; ++pt_grid_x)
    {
        for (int pt_grid_y = 0; pt_grid_y < voxel_grid_dim_y; ++pt_grid_y)
        {
            for (int pt_grid_z = 0; pt_grid_z < voxel_grid_dim_z; ++pt_grid_z)
            {

                // voxel in world coor
                float pt_base_x = voxel_grid_origin_x + pt_grid_x * voxel_size;
                float pt_base_y = voxel_grid_origin_y + pt_grid_y * voxel_size;
                float pt_base_z = voxel_grid_origin_z + pt_grid_z * voxel_size;

                // world coordinates to camera coordinates
                float tmp_pt[3] = {0};
                tmp_pt[0] = pt_base_x - cam2base[0 * 4 + 3];
                tmp_pt[1] = pt_base_y - cam2base[1 * 4 + 3];
                tmp_pt[2] = pt_base_z - cam2base[2 * 4 + 3];
                float pt_cam_x = cam2base[0 * 4 + 0] * tmp_pt[0] + cam2base[1 * 4 + 0] * tmp_pt[1] + cam2base[2 * 4 + 0] * tmp_pt[2];
                float pt_cam_y = cam2base[0 * 4 + 1] * tmp_pt[0] + cam2base[1 * 4 + 1] * tmp_pt[1] + cam2base[2 * 4 + 1] * tmp_pt[2];
                float pt_cam_z = cam2base[0 * 4 + 2] * tmp_pt[0] + cam2base[1 * 4 + 2] * tmp_pt[1] + cam2base[2 * 4 + 2] * tmp_pt[2];

                if (pt_cam_z <= 0)
                    continue;

                // camera coor to pixel coor
                int pt_pix_x = roundf(cam_K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + cam_K[0 * 3 + 2]);
                int pt_pix_y = roundf(cam_K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + cam_K[1 * 3 + 2]);
                if (pt_pix_x < 0 || pt_pix_x >= im_width || pt_pix_y < 0 || pt_pix_y >= im_height)
                    continue;

                float depth_val = float(depth_im[pt_pix_y * im_width + pt_pix_x]) / 1000.0;
                // cout << depth_val << " ";

                float diff = depth_val - pt_cam_z; // Signed Distance Func

                if (diff <= -trunc_margin) // 截断-1隐势面后的数据
                    continue;

                // Integrate
                // 加权方法：1.可直接设1，用最新的； 2.可根据角度加权，角度小则权重大
                int volume_idx = pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x;
                float dist = fmin(1.0f, diff / trunc_margin);
                float weight_old = voxel_grid_weight[volume_idx];
                float weight_new = weight_old + 1.0f;
                voxel_grid_weight[volume_idx] = weight_new;
                voxel_grid_TSDF[volume_idx] = (voxel_grid_TSDF[volume_idx] * weight_old + dist) / weight_new;
            }
        }
    }
}

// Volume is aligned to the coor of base frame
// TSDF voxel volume (5m * 5m * 5m at 1cm resolution)
int main(int argc, char *argv[])
{
    string cam_K_file = "/home/tan/projects/mapping/data/camera-intrinsics.txt";
    string data_path = "/home/tan/projects/mapping/data/tmp";

    // string cam_K_file = "/userdata/camera-intrinsics.txt";
    // string data_path = "/userdata/tmp";
    int base_frame_idx = 0;
    int first_frame_idx = 0;
    float num_frames = 36;

    float cam_K[3 * 3];
    float base2world[4 * 4];
    float cam2base[4 * 4];
    float cam2world[4 * 4];

    int im_width = 480;
    int im_height = 640;
    unsigned short depth_im[im_height * im_width];

    // TSDF起始点
    float factor = 2.5;                // 1x 分辨率， 立方关系
    float voxel_grid_origin_x = -5.0f; // location of voxel grid origin in base frame camera coor
    float voxel_grid_origin_y = -2.0f;
    float voxel_grid_origin_z = -4.5f;
    // TSDF voxel num 10e6
    int voxel_grid_dim_x = int(100 * factor);
    int voxel_grid_dim_y = int(40 * factor);
    int voxel_grid_dim_z = int(90 * factor);
    float voxel_size = 0.1f / factor;    // voxel resolution
    float trunc_margin = voxel_size * 4; //

    // Read camera intrinsics
    vector<float> cam_K_vec = LoadMatrixFromFile(cam_K_file, 3, 3);
    copy(cam_K_vec.begin(), cam_K_vec.end(), cam_K);

    // Read base pose
    ostringstream base_frame_prefix;
    base_frame_prefix << std::setw(4) << std::setfill('0') << base_frame_idx;
    string base2world_file = data_path + "/" + base_frame_prefix.str() + ".txt";
    vector<float> base2world_vec = LoadMatrixFromFile(base2world_file, 4, 4);
    copy(base2world_vec.begin(), base2world_vec.end(), base2world);

    // invert base pose to get world-to-base transform
    float base2world_inv[16] = {0};
    invert_matrix(base2world, base2world_inv); // only one-time

    // initial voxel grid
    int voxel_volumn = voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z;
    float *voxel_grid_TSDF = new float[voxel_volumn];
    float *voxel_grid_weight = new float[voxel_volumn];
    for (int i = 0; i < voxel_volumn; i++)
    {
        voxel_grid_TSDF[i] = 1.0f;
        voxel_grid_weight[i] = 0.0f;
    }

    // Loop frame and integrade to TSDF voxel grid
    float yaw = 0;
    for (int frame_idx = first_frame_idx; frame_idx < first_frame_idx + (int)num_frames; frame_idx++)
    {
        auto start = system_clock::now();
        ostringstream curr_frame_prefix;
        curr_frame_prefix << setw(4) << setfill('0') << frame_idx;
        // Read current frame depth
        // string depth_im_file = data_path + "/depth" + curr_frame_prefix.str() + ".png";
        // ReadDepth(depth_im_file, im_height, im_width, depth_im);

        string depth_im_file1 = data_path + "/depth" + curr_frame_prefix.str();
        // WriteDepthBin(depth_im_file1, im_height, im_width, depth_im);

        ReadDepthBin(depth_im_file1, im_height, im_width, depth_im);
        cout << "Fusing: " << depth_im_file1 << endl;

        // read pose
        string cam2world_file = data_path + "/" + curr_frame_prefix.str() + ".txt";
        vector<float> cam2world_vec = LoadMatrixFromFile(cam2world_file, 4, 4);
        // vector<float> cam2world_vec;
        // MatrixFromYawPitchRoll(yaw+frame_idx*10.0, 0.0, 0.0, cam2world_vec);

        copy(cam2world_vec.begin(), cam2world_vec.end(), cam2world);

        // camera coor to world coor
        multiply_matrix(base2world_inv, cam2world, cam2base);

        // Incremental Integrate
        Integrate(cam_K, cam2base, depth_im,
                  im_height, im_width,
                  voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
                  voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
                  voxel_size, trunc_margin, voxel_grid_TSDF, voxel_grid_weight);

        auto end = system_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        cout << "spend " << double(duration.count()) * microseconds::period::num / microseconds::period::den << " second" << endl;
    }

    cout << "Saving surface point cloud (tsdf.ply)..." << endl;

    SaveVoxelGrid2SurfacePointCloud("tsdf.ply", voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
                                    voxel_size, voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
                                    voxel_grid_TSDF, voxel_grid_weight, 0.25, 2.0f); // 后两超参： 到隐势面的距离阈值（-1,1), 看到voxel的次数; 即(-0.4,0.4),3次

    return 0;
}
