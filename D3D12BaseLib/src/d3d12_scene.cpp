#include "d3d12_scene.h"

void Model::draw(CommandList &cmdList, const Scene &scene) const {
	const SizeType numMeshes = meshes.size();
	for (int i = 0; i < numMeshes; ++i) {
		const Mesh &mesh = meshes[i];

		// TODO: update uniform buffers
		// we'll need to read the material data for the mesh's material.
		// Materials are global and reside in the scene object.



		cmdList->DrawIndexedInstanced(mesh.numIndices, 1, mesh.indexOffset, 0, 0);
	}
}

void Scene::draw(CommandList &cmdList) const {
	// TODO: any lights/cameras/other objects global to the scene
	// that need to go as data in the shader, should go here
	// before recursing the node tree.

	const SizeType numNodes = nodes.size();
	for (int i = 0; i < numNodes; ++i) {
		drawNodeImpl(nodes[i], cmdList, *this);
	}
}

void Scene::drawNodeImpl(Node *node, CommandList &cmdList, const Scene &scene) const {
	node->draw(cmdList, scene);

	const SizeType numChildren = node->children.size();
	for (int i = 0; i < numChildren; ++i) {
		drawNodeImpl(nodes[node->children[i]], cmdList, scene);
	}
}
