/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  Timotej Halenár <xhalen00@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    26.11.2024
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{
}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    // Suggested approach to tackle this problem is to add new method to
    // this class. This method will call itself to process the children.
    // It is also strongly suggested to first implement Octree as sequential
    // code and only when that works add OpenMP tasks to achieve parallelism.

    Vec3_t<float> cubeOffset = (0, 0, 0);

    return processCube(mGridSize, cubeOffset, field);
}

unsigned TreeMeshBuilder::processCube(size_t cubeSize, Vec3_t<float> cubeOffset, const ParametricScalarField &field)
{
    unsigned totalTriangles = 0;

    // std::cout << "size: " << cubeSize << "offset: " << cubeOffset.x << ", " << cubeOffset.y << ", " << cubeOffset.z << std::endl;

    if (cubeSize == 1)
    {
        // std::cout << "offset: " << cubeOffset.x << ", " << cubeOffset.y << ", " << cubeOffset.z << std::endl;
        return buildCube(cubeOffset, field);
    }

    // TODO
    //  check if surface passes through here
    //  if not, return
    //  if yes, divide into 8 more cubes

    size_t halfSize = cubeSize / 2;
    if (halfSize < 1)
    {
        throw "WrongSize";
    }

    float floatHalf = (float)halfSize;

    float x = cubeOffset.x;
    float y = cubeOffset.y;
    float z = cubeOffset.z;

    Vec3_t<float> middleCube(x + floatHalf, y + floatHalf, z + floatHalf);
    CubeCornerVerts_t cubeCorners;
    transformCubeVertices(middleCube, sc_vertexNormPos, cubeCorners);

    float midVal = evaluateFieldAt(cubeCorners[0], field);

    if (midVal > mIsoLevel + (sqrt(3) / 2.0f) * floatHalf)
    {
        return 0;
    }

    Vec3_t<float> cubeOffset0(x, y, z);
    Vec3_t<float> cubeOffset1(x + floatHalf, y, z);
    Vec3_t<float> cubeOffset2(x, y + floatHalf, z);
    Vec3_t<float> cubeOffset3(x + floatHalf, y + floatHalf, z);
    Vec3_t<float> cubeOffset4(x, y, z + floatHalf);
    Vec3_t<float> cubeOffset5(x + floatHalf, y, z + floatHalf);
    Vec3_t<float> cubeOffset6(x, y + floatHalf, z + floatHalf);
    Vec3_t<float> cubeOffset7(x + floatHalf, y + floatHalf, z + floatHalf);

    totalTriangles += processCube(halfSize, cubeOffset0, field);
    totalTriangles += processCube(halfSize, cubeOffset1, field);
    totalTriangles += processCube(halfSize, cubeOffset2, field);
    totalTriangles += processCube(halfSize, cubeOffset3, field);
    totalTriangles += processCube(halfSize, cubeOffset4, field);
    totalTriangles += processCube(halfSize, cubeOffset5, field);
    totalTriangles += processCube(halfSize, cubeOffset6, field);
    totalTriangles += processCube(halfSize, cubeOffset7, field);

    return totalTriangles;
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    // NOTE: This method is called from "buildCube(...)"!

    // 1. Store pointer to and number of 3D points in the field
    //    (to avoid "data()" and "size()" call in the loop).
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    // 2. Find minimum square distance from points "pos" to any point in the
    //    field.
    for (unsigned i = 0; i < count; ++i)
    {
        float distanceSquared = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
        distanceSquared += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
        distanceSquared += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);

        // Comparing squares instead of real distance to avoid unnecessary
        // "sqrt"s in the loop.
        value = std::min(value, distanceSquared);
    }

    // 3. Finally take square root of the minimal square distance to get the real distance
    return sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    mTriangles.push_back(triangle);
}
