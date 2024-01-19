#pragma once

#include "math/dar_math.h"
#include "utils/defines.h"

using SkeletonId = SizeType;
using AnimationId = SizeType;

#define INVALID_SKELETON_ID SizeType(-1)
#define INVALID_ANIMATION_ID SizeType(-1)


struct Joint {
	String name;
	Mat4 invBindPose;
	uint8_t parent;
};

struct Skeleton {
	SkeletonId id;
	Vector<Joint> joints;
};

struct JointPose {
	Quaternion rotation = Quaternion();
	Vec3 translation = Vec3(0.f);
	float scale = 1.f;
};

struct SkeletonPose {
	SkeletonId skeletonId;
	Vector<JointPose> jointPoses;
};

struct AnimationClip {
	using Sample = SkeletonPose;

	Vector<Sample> samples;
	float animationLengthMs = 0.f;
};

struct AnimationManager {
	Vector<Skeleton> skeletons;
	Map<String, AnimationClip> nameToAnimation;
};