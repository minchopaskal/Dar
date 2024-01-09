#include "common.hlsli"

[numthreads(1024, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint3 tid : SV_DispatchThreadID, uint3 localTID : SV_GroupThreadID, uint groupIndex : SV_GroupIndex)
{
	const int index = tid.x;

	StructuredBuffer<int> srcBuffer = ResourceDescriptorHeap[0];
	RWStructuredBuffer<int> dstBuffer = ResourceDescriptorHeap[1];

	if (index == 0 || index == 3) {
		dstBuffer[index] = 1;
	} else if (index < 4) {
		dstBuffer[index] = 0;
	} else {
		dstBuffer[index] = srcBuffer[index] + 10;
	}
}
