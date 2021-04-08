// scene/demo-specific variables go here
var sceneIsDynamic = true;
var camFlightSpeed = 60;
var GLTF_Model_Geometry, GLTF_Model_Material, GLTF_Model_Mesh;
var albedoMap, emissiveMap, metallicRoughnessMap, normalMap;
var animationTimer = 0.0;
var animationAxis = new THREE.Vector3(0, 0, 1);
var modelMesh;
var modelScale = 1.0;
var modelPositionOffset = new THREE.Vector3();
var total_number_of_triangles = 0;
var triangle_array;
var triangleMaterialMarkers = [];
var pathTracingMaterialList = [];
var uniqueMaterialTextures = [];
var meshList = [];
var geoList = [];
var triangleDataTexture;
var aabb_array;
var aabbDataTexture;
var totalWork;
var vp0 = new THREE.Vector3();
var vp1 = new THREE.Vector3();
var vp2 = new THREE.Vector3();
var vn0 = new THREE.Vector3();
var vn1 = new THREE.Vector3();
var vn2 = new THREE.Vector3();
var vt0 = new THREE.Vector2();
var vt1 = new THREE.Vector2();
var vt2 = new THREE.Vector2();



function MaterialObject() {
// a list of material types and their corresponding numbers are found in the 'pathTracingCommon.js' file
        this.type = 1; // default is '1': diffuse type 		
        this.albedoTextureID = -1; // which diffuse map to use for model's color / '-1' = no textures are used
        this.color = new THREE.Color(1.0, 1.0, 1.0); // takes on different meanings, depending on 'type' above
        this.roughness = 0.0; // 0.0 to 1.0 range, perfectly smooth to extremely rough
        this.metalness = 0.0; // 0.0 to 1.0 range, usually either 0 or 1, either non-metal or metal
        this.opacity = 1.0;   // 0.0 to 1.0 range, fully transparent to fully opaque
        this.refractiveIndex = 1.0; // 1.0=air, 1.33=water, 1.4=clearCoat, 1.5=glass, etc.
}


function load_GLTF_Model() {

        var gltfLoader = new THREE.GLTFLoader();

        modelScale = 10.0; // good scale size for DamagedHelmet model
        
        gltfLoader.load("models/DamagedHelmet.gltf", function( meshGroup ) { // Triangles: 15,452
                
                if (meshGroup.scene)
                        meshGroup = meshGroup.scene;
                        
                let totalTriangleCount = 0;

                meshGroup.traverse( function ( child ) {

                        if ( child.isMesh ) {
                                let mat = new MaterialObject();

                                // note: '4' means clearCoat material over diffuse material.  the clearCoat portion will have an IoR of around 1.4
                                let materialType = child.material.opacity < 1 ? 2 : 4; // 2 = glossy transparent, 4 = glossy opaque
                                mat.type = materialType;
                                mat.color.copy(child.material.color);
                                mat.roughness = child.material.roughness || 0.0;
                                mat.metalness = child.material.metalness || 0.0;
                                mat.opacity = child.material.opacity || 1.0;
                                mat.refractiveIndex = materialType == 4 ? 1.4 : 1.52; // IoR of clearCoat = 1.4, glass = 1.5
                                pathTracingMaterialList.push(mat);
                                totalTriangleCount += child.geometry.index.count / 3;
                                triangleMaterialMarkers.push(totalTriangleCount);
                                meshList.push(child);
                        }
                } );

                modelMesh = meshList[0].clone();

                for (let i = 0; i < meshList.length; i++) {
                        geoList.push(meshList[i].geometry);
                }

                modelMesh.geometry = THREE.BufferGeometryUtils.mergeBufferGeometries(geoList);
                
                if (modelMesh.geometry.index)
                        modelMesh.geometry = modelMesh.geometry.toNonIndexed();

                modelMesh.geometry.center();

                
                // albedo map
                if (meshList[0].material.map != undefined)
                        albedoMap = meshList[0].material.map;

                // emissive map
                if (meshList[0].material.emissiveMap != undefined)
                        emissiveMap = meshList[0].material.emissiveMap;

                // metallicRoughness map
                if (meshList[0].material.roughnessMap != undefined)
                        metallicRoughnessMap = meshList[0].material.roughnessMap;
                
                // normal map
                if (meshList[0].material.normalMap != undefined)
                        normalMap = meshList[0].material.normalMap;
                
                
                for (let i = 0; i < meshList.length; i++) {
                        if (meshList[i].material.map != undefined) {
                                pathTracingMaterialList[i].textureID = 0;	
                        }				
                }
                
                // now that the models have been loaded, we can init
                init();

        });

} // end function load_GLTF_Model()



// called automatically from within initTHREEjs() function
function initSceneData() {
        
        // scene/demo-specific three.js objects setup goes here
        EPS_intersect = mouseControl ? 0.01 : 1.0; // less precision on mobile

        GLTF_Model_Geometry = new THREE.BoxGeometry(1,1,1);
        GLTF_Model_Material = new THREE.MeshPhysicalMaterial( {
                color: new THREE.Color(0.95, 0.95, 0.95), //RGB, ranging from 0.0 - 1.0
                roughness: 1.0 // ideal Diffuse material	
        } );
        
        GLTF_Model_Mesh = new THREE.Mesh(GLTF_Model_Geometry, GLTF_Model_Material);
        pathTracingScene.add(GLTF_Model_Mesh);
        GLTF_Model_Mesh.visible = false;
        //GLTF_Model_Mesh.rotation.set(-Math.PI * 0.5, 0, 0); // for riggedFigure model
        GLTF_Model_Mesh.rotation.set(Math.PI * 0.5, 0, 0); // for DamagedHelmet model
        //GLTF_Model_Mesh.position.set(0, 40, 0);
        GLTF_Model_Mesh.updateMatrixWorld(true); // 'true' forces immediate matrix update

        // set camera's field of view
        worldCamera.fov = 60;
        focusDistance = 124.0;

        // position and orient camera
        cameraControlsObject.position.set(0, 40, 100);
        // look slightly downward
        //cameraControlsPitchObject.rotation.x = -0.2;
        

        total_number_of_triangles = modelMesh.geometry.attributes.position.array.length / 9;
        console.log("Triangle count:" + total_number_of_triangles);

        totalWork = new Uint32Array(total_number_of_triangles);

        triangle_array = new Float32Array(2048 * 2048 * 4);
        // 2048 = width of texture, 2048 = height of texture, 4 = r,g,b, and a components
        
        aabb_array = new Float32Array(2048 * 2048 * 4);
        // 2048 = width of texture, 2048 = height of texture, 4 = r,g,b, and a components

        
        var triangle_b_box_min = new THREE.Vector3();
        var triangle_b_box_max = new THREE.Vector3();
        var triangle_b_box_centroid = new THREE.Vector3();
        

        var vpa = new Float32Array(modelMesh.geometry.attributes.position.array);
        var vna = new Float32Array(modelMesh.geometry.attributes.normal.array);
        var vta = null;
        var modelHasUVs = false;
        if (modelMesh.geometry.attributes.uv !== undefined) {
                vta = new Float32Array(modelMesh.geometry.attributes.uv.array);
                modelHasUVs = true;
        }
                
        var materialNumber = 0;

        for (let i = 0; i < total_number_of_triangles; i++) {
        
                triangle_b_box_min.set(Infinity, Infinity, Infinity);
                triangle_b_box_max.set(-Infinity, -Infinity, -Infinity);

                for (let j = 0; j < pathTracingMaterialList.length; j++) {
                        if (i < triangleMaterialMarkers[j]) {
                                materialNumber = j;
                                break;
                        }
                }

                // record vertex texture coordinates (UVs)
                if (modelHasUVs) {
                        vt0.set( vta[6 * i + 0], vta[6 * i + 1] );
                        vt1.set( vta[6 * i + 2], vta[6 * i + 3] );
                        vt2.set( vta[6 * i + 4], vta[6 * i + 5] );
                }
                else {
                        vt0.set( -1, -1 );
                        vt1.set( -1, -1 );
                        vt2.set( -1, -1 );
                }
                
                // record vertex normals
                vn0.set( vna[9 * i + 0], vna[9 * i + 1], vna[9 * i + 2] ).normalize();
                vn1.set( vna[9 * i + 3], vna[9 * i + 4], vna[9 * i + 5] ).normalize();
                vn2.set( vna[9 * i + 6], vna[9 * i + 7], vna[9 * i + 8] ).normalize();
                
                // record vertex positions
                vp0.set( vpa[9 * i + 0], vpa[9 * i + 1], vpa[9 * i + 2] );
                vp1.set( vpa[9 * i + 3], vpa[9 * i + 4], vpa[9 * i + 5] );
                vp2.set( vpa[9 * i + 6], vpa[9 * i + 7], vpa[9 * i + 8] );

                vp0.multiplyScalar(modelScale);
                vp1.multiplyScalar(modelScale);
                vp2.multiplyScalar(modelScale);

                vp0.add(modelPositionOffset);
                vp1.add(modelPositionOffset);
                vp2.add(modelPositionOffset);

                //slot 0
                triangle_array[32 * i +  0] = vp0.x; // r or x
                triangle_array[32 * i +  1] = vp0.y; // g or y 
                triangle_array[32 * i +  2] = vp0.z; // b or z
                triangle_array[32 * i +  3] = vp1.x; // a or w

                //slot 1
                triangle_array[32 * i +  4] = vp1.y; // r or x
                triangle_array[32 * i +  5] = vp1.z; // g or y
                triangle_array[32 * i +  6] = vp2.x; // b or z
                triangle_array[32 * i +  7] = vp2.y; // a or w

                //slot 2
                triangle_array[32 * i +  8] = vp2.z; // r or x
                triangle_array[32 * i +  9] = vn0.x; // g or y
                triangle_array[32 * i + 10] = vn0.y; // b or z
                triangle_array[32 * i + 11] = vn0.z; // a or w

                //slot 3
                triangle_array[32 * i + 12] = vn1.x; // r or x
                triangle_array[32 * i + 13] = vn1.y; // g or y
                triangle_array[32 * i + 14] = vn1.z; // b or z
                triangle_array[32 * i + 15] = vn2.x; // a or w

                //slot 4
                triangle_array[32 * i + 16] = vn2.y; // r or x
                triangle_array[32 * i + 17] = vn2.z; // g or y
                triangle_array[32 * i + 18] = vt0.x; // b or z
                triangle_array[32 * i + 19] = vt0.y; // a or w

                //slot 5
                triangle_array[32 * i + 20] = vt1.x; // r or x
                triangle_array[32 * i + 21] = vt1.y; // g or y
                triangle_array[32 * i + 22] = vt2.x; // b or z
                triangle_array[32 * i + 23] = vt2.y; // a or w

                // the remaining slots are used for PBR material properties

                //slot 6
                triangle_array[32 * i + 24] = pathTracingMaterialList[materialNumber].type; // r or x 
                triangle_array[32 * i + 25] = pathTracingMaterialList[materialNumber].color.r; // g or y
                triangle_array[32 * i + 26] = pathTracingMaterialList[materialNumber].color.g; // b or z
                triangle_array[32 * i + 27] = pathTracingMaterialList[materialNumber].color.b; // a or w

                //slot 7
                triangle_array[32 * i + 28] = pathTracingMaterialList[materialNumber].albedoTextureID; // r or x
                triangle_array[32 * i + 29] = 0; // g or y
                triangle_array[32 * i + 30] = 0; // b or z
                triangle_array[32 * i + 31] = 0; // a or w

                triangle_b_box_min.copy(triangle_b_box_min.min(vp0));
                triangle_b_box_max.copy(triangle_b_box_max.max(vp0));
                triangle_b_box_min.copy(triangle_b_box_min.min(vp1));
                triangle_b_box_max.copy(triangle_b_box_max.max(vp1));
                triangle_b_box_min.copy(triangle_b_box_min.min(vp2));
                triangle_b_box_max.copy(triangle_b_box_max.max(vp2));

                triangle_b_box_centroid.set((triangle_b_box_min.x + triangle_b_box_max.x) * 0.5,
                                            (triangle_b_box_min.y + triangle_b_box_max.y) * 0.5,
                                            (triangle_b_box_min.z + triangle_b_box_max.z) * 0.5);

                aabb_array[9 * i + 0] = triangle_b_box_min.x;
                aabb_array[9 * i + 1] = triangle_b_box_min.y;
                aabb_array[9 * i + 2] = triangle_b_box_min.z;
                aabb_array[9 * i + 3] = triangle_b_box_max.x;
                aabb_array[9 * i + 4] = triangle_b_box_max.y;
                aabb_array[9 * i + 5] = triangle_b_box_max.z;
                aabb_array[9 * i + 6] = triangle_b_box_centroid.x;
                aabb_array[9 * i + 7] = triangle_b_box_centroid.y;
                aabb_array[9 * i + 8] = triangle_b_box_centroid.z;

                totalWork[i] = i;
        }


        // Build the BVH acceleration structure, which places a bounding box ('root' of the tree) around all of the 
        // triangles of the entire mesh, then subdivides each box into 2 smaller boxes.  It continues until it reaches 1 triangle,
        // which it then designates as a 'leaf'
        BVH_Build_Iterative(totalWork, aabb_array);
        

        triangleDataTexture = new THREE.DataTexture(triangle_array,
                2048,
                2048,
                THREE.RGBAFormat,
                THREE.FloatType,
                THREE.Texture.DEFAULT_MAPPING,
                THREE.ClampToEdgeWrapping,
                THREE.ClampToEdgeWrapping,
                THREE.NearestFilter,
                THREE.NearestFilter,
                1,
                THREE.LinearEncoding);

        triangleDataTexture.flipY = false;
        triangleDataTexture.generateMipmaps = false;
        triangleDataTexture.needsUpdate = true;

        aabbDataTexture = new THREE.DataTexture(aabb_array,
                2048,
                2048,
                THREE.RGBAFormat,
                THREE.FloatType,
                THREE.Texture.DEFAULT_MAPPING,
                THREE.ClampToEdgeWrapping,
                THREE.ClampToEdgeWrapping,
                THREE.NearestFilter,
                THREE.NearestFilter,
                1,
                THREE.LinearEncoding);

        aabbDataTexture.flipY = false;
        aabbDataTexture.generateMipmaps = false;
        aabbDataTexture.needsUpdate = true;

} // end function initSceneData()



// called automatically from within initTHREEjs() function
function initPathTracingShaders() {
 
        // scene/demo-specific uniforms go here
        pathTracingUniforms.tPreviousTexture = { type: "t", value: screenCopyRenderTarget.texture };
	pathTracingUniforms.tTriangleTexture = { type: "t", value: triangleDataTexture };
	pathTracingUniforms.tAABBTexture = { type: "t", value: aabbDataTexture };
	pathTracingUniforms.tAlbedoMap = { type: "t", value: albedoMap };
	pathTracingUniforms.tEmissiveMap = { type: "t", value: emissiveMap };
	pathTracingUniforms.tMetallicRoughnessMap = { type: "t", value: metallicRoughnessMap };
	pathTracingUniforms.tNormalMap = { type: "t", value: normalMap };
	pathTracingUniforms.uGLTF_Model_Position = { type: "v3", value: new THREE.Vector3() };
	pathTracingUniforms.uGLTF_Model_InvMatrix = { type: "m4", value: new THREE.Matrix4() };

        pathTracingDefines = {
        	//NUMBER_OF_TRIANGLES: total_number_of_triangles
        };

        // load vertex and fragment shader files that are used in the pathTracing material, mesh and scene
        fileLoader.load('shaders/common_PathTracing_Vertex.glsl', function (shaderText) {
                pathTracingVertexShader = shaderText;

                createPathTracingMaterial();
        });

} // end function initPathTracingShaders()


// called automatically from within initPathTracingShaders() function above
function createPathTracingMaterial() {

        fileLoader.load('shaders/BVH_Animated_Model_Fragment.glsl', function (shaderText) {
                
                pathTracingFragmentShader = shaderText;

                pathTracingMaterial = new THREE.ShaderMaterial({
                        uniforms: pathTracingUniforms,
                        defines: pathTracingDefines,
                        vertexShader: pathTracingVertexShader,
                        fragmentShader: pathTracingFragmentShader,
                        depthTest: false,
                        depthWrite: false
                });

                pathTracingMesh = new THREE.Mesh(pathTracingGeometry, pathTracingMaterial);
                pathTracingScene.add(pathTracingMesh);

                // the following keeps the large scene ShaderMaterial quad right in front 
                //   of the camera at all times. This is necessary because without it, the scene 
                //   quad will fall out of view and get clipped when the camera rotates past 180 degrees.
                worldCamera.add(pathTracingMesh);
                
        });

} // end function createPathTracingMaterial()



// called automatically from within the animate() function
function updateVariablesAndUniforms() {

        // GLTF Model
        animationTimer += 1.0 * frameTime;
        GLTF_Model_Mesh.rotateOnWorldAxis(animationAxis, 0.1 * frameTime);
        GLTF_Model_Mesh.position.set(100 * Math.cos(animationTimer * 0.05), 20 * (Math.sin(animationTimer * 0.2) + 1.5), 120 * Math.sin(animationTimer * 0.05) - 50 );
        ///GLTF_Model_Mesh.position.set(0,40,80);

        GLTF_Model_Mesh.updateMatrixWorld(true); // 'true' forces immediate matrix update
        pathTracingUniforms.uGLTF_Model_Position.value.copy(GLTF_Model_Mesh.position);
        pathTracingUniforms.uGLTF_Model_InvMatrix.value.copy( GLTF_Model_Mesh.matrixWorld ).invert();
        
        // INFO
        cameraInfoElement.innerHTML = "FOV: " + worldCamera.fov + " / Aperture: " + apertureSize.toFixed(2) +
                " / FocusDistance: " + focusDistance + "<br>" + "Samples: " + sampleCounter;

} // end function updateUniforms()



load_GLTF_Model(); // load model, init app, and start animating
