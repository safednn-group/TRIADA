import os
import subprocess
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

REPO_ROOT = Path(__file__).resolve().parent


def get_install_requirements():
    requirements_path = REPO_ROOT / "requirements.txt"
    return [
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def get_git_commit_number():
    if not (REPO_ROOT / ".git").exists():
        return "0000000"

    result = subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name="%s.%s" % (module, name),
        sources=[os.path.join(*module.split("."), src) for src in sources],
    )
    return cuda_ext


def write_version_to_file(version, target_file):
    with target_file.open("w") as f:
        print('__version__ = "%s"' % version, file=f)


if __name__ == "__main__":
    version = "0.6.0+%s" % get_git_commit_number()
    write_version_to_file(version, REPO_ROOT / "pcdet" / "version.py")

    setup(
        name="pcdet",
        version=version,
        description="TRIADA reproduction code based on OpenPCDet",
        # torch and spconv are installed separately because their package names
        # depend on the selected CUDA version.
        install_requires=get_install_requirements(),
        author="Shaoshuai Shi",
        author_email="shaoshuaics@gmail.com",
        license="Apache License 2.0",
        packages=find_packages(exclude=["tools", "data", "output"]),
        cmdclass={
            "build_ext": BuildExtension,
        },
        ext_modules=[
            make_cuda_ext(
                name="iou3d_nms_cuda",
                module="pcdet.ops.iou3d_nms",
                sources=[
                    "src/iou3d_cpu.cpp",
                    "src/iou3d_nms_api.cpp",
                    "src/iou3d_nms.cpp",
                    "src/iou3d_nms_kernel.cu",
                ],
            ),
            make_cuda_ext(
                name="roiaware_pool3d_cuda",
                module="pcdet.ops.roiaware_pool3d",
                sources=[
                    "src/roiaware_pool3d.cpp",
                    "src/roiaware_pool3d_kernel.cu",
                ],
            ),
            make_cuda_ext(
                name="pointnet2_stack_cuda",
                module="pcdet.ops.pointnet2.pointnet2_stack",
                sources=[
                    "src/pointnet2_api.cpp",
                    "src/ball_query.cpp",
                    "src/ball_query_gpu.cu",
                    "src/group_points.cpp",
                    "src/group_points_gpu.cu",
                    "src/sampling.cpp",
                    "src/sampling_gpu.cu",
                    "src/interpolate.cpp",
                    "src/interpolate_gpu.cu",
                ],
            ),
        ],
    )
