#include "d3d12_scene.h"

void Model::draw(CommandList &cmdList, const Scene &scene) const {
	const SizeType numMeshes = meshes.size();
	for (int i = 0; i < numMeshes; ++i) {
		const Mesh &mesh = meshes[i];

		// TODO: update uniform buffers
		cmdList->DrawIndexedInstanced(mesh.numIndices, 1, 0, 0, 0);
	}
}

void Scene::drawImpl(Node *node, CommandList &cmdList, const Scene &scene) const {
	node->draw(cmdList, scene);

	const SizeType numChildren = node->children.size();
	for (int i = 0; i < numChildren; ++i) {
		drawImpl(nodes[node->children[i]], cmdList, scene);
	}
}
