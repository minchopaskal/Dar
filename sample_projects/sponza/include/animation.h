#pragma once

#include "math/dar_math.h"
#include "utils/defines.h"

using AnimationId = SizeType;
using SkeletonId = SizeType;
using JointId = u8;

const SkeletonId INVALID_SKELETON_ID = SizeType(-1);
const AnimationId INVALID_ANIMATION_ID = SizeType(-1);
const u8 INVALID_JOINT_ID = u8(-1);

struct Joint {
	String name = "";

	Mat4 inverseBindPose = Mat4(1.f);
	JointId parentJoint = INVALID_JOINT_ID;
};

struct JointPose {
	Quat rotation = Quat::identity();
	JointId jointHandle = INVALID_JOINT_ID;

	bool isIdentity() const {
		return rotation == Quat::identity();
	}
};

struct AnimationSample {
	JointPose *jointPoses = nullptr;
};

struct AnimationSkeleton {
	Joint *joints = nullptr;
	u8 numJoints = 0; ///< Maximum of 255 joints per skeleton
};

struct AnimationClip {
	AnimationSample *samples = nullptr;
	String name = "";
	SkeletonId skeleton = INVALID_SKELETON_ID;
	u32 frameCount = 0;
	float duration = 0.f; // seconds
};

struct AnimationManager {
	Vector<AnimationClip> animations;
	Vector<AnimationSkeleton> skeletons;
};
