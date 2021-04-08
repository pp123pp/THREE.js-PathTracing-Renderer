precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D tTriangleTexture;
uniform sampler2D tAABBTexture;
uniform sampler2D tHDRTexture;
//uniform sampler2D tMarbleTexture;

#include <pathtracing_uniforms_and_defines>

uniform vec3 uMaterialColor;
uniform float uHDRI_Exposure;
uniform float uRoughness;
uniform int uMaterialType;

//float InvTextureWidth = 0.000244140625; // (1 / 4096 texture width)
//float InvTextureWidth = 0.00048828125;  // (1 / 2048 texture width)
//float InvTextureWidth = 0.0009765625;   // (1 / 1024 texture width)

#define INV_TEXTURE_WIDTH 0.00048828125

// the following directions pointing at the sun were found by trial and error using the large red metal cylinder as a guide
#define SUN_DIRECTION normalize(vec3(-0.555, 1.0, 0.205)) // use this vec3 for the symmetrical_garden_2k.hdr environment
//#define SUN_DIRECTION normalize(vec3(0.54, 1.0, -0.595)) // use this vec3 for the kiara_5_noon_2k.hdr environment

#define N_SPHERES 4
#define N_BOXES 2

//-----------------------------------------------------------------------

struct Ray { vec3 origin; vec3 direction; };
struct Sphere { float radius; vec3 position; vec3 emission; vec3 color; int type; };
struct OpenCylinder { vec3 p0; vec3 p1; float radius; vec3 emission; vec3 color; int type; };
struct Box { vec3 minCorner; vec3 maxCorner; vec3 emission; vec3 color; int type; };
struct Intersection { vec3 normal; vec3 emission; vec3 color; vec2 uv; int type; bool isModel; };

Sphere spheres[N_SPHERES];
OpenCylinder openCylinders[1];
Box boxes[N_BOXES];


#include <pathtracing_random_functions>

#include <pathtracing_calc_fresnel_reflectance>

#include <pathtracing_sphere_intersect>

#include <pathtracing_box_intersect>

#include <pathtracing_boundingbox_intersect>

#include <pathtracing_bvhTriangle_intersect>

//#include <pathtracing_bvhDoubleSidedTriangle_intersect>

#include <pathtracing_opencylinder_intersect>
	

vec2 stackLevels[28];

struct BoxNode
{
	vec4 data0; // corresponds to .x: idTriangle, .y: aabbMin.x, .z: aabbMin.y, .w: aabbMin.z
	vec4 data1; // corresponds to .x: idRightChild .y: aabbMax.x, .z: aabbMax.y, .w: aabbMax.z
};

BoxNode GetBoxNode(const in float i)
{
	// each bounding box's data is encoded in 2 rgba(or xyzw) texture slots 
	float iX2 = (i * 2.0);
	// (iX2 + 0.0) corresponds to .x: idTriangle, .y: aabbMin.x, .z: aabbMin.y, .w: aabbMin.z 
	// (iX2 + 1.0) corresponds to .x: idRightChild .y: aabbMax.x, .z: aabbMax.y, .w: aabbMax.z 

	ivec2 uv0 = ivec2( mod(iX2 + 0.0, 2048.0), (iX2 + 0.0) * INV_TEXTURE_WIDTH ); // data0
	ivec2 uv1 = ivec2( mod(iX2 + 1.0, 2048.0), (iX2 + 1.0) * INV_TEXTURE_WIDTH ); // data1
	
	return BoxNode( texelFetch(tAABBTexture, uv0, 0), texelFetch(tAABBTexture, uv1, 0) );
}


//------------------------------------------------------------------------------------
float SceneIntersect( Ray r, inout Intersection intersec, out bool finalIsRayExiting )
//------------------------------------------------------------------------------------
{
        BoxNode currentBoxNode, nodeA, nodeB, tmpNode;
	
	vec4 vd0, vd1, vd2, vd3, vd4, vd5, vd6, vd7;

	vec3 inverseDir = 1.0 / r.direction;
	vec3 normal;

	vec2 currentStackData, stackDataA, stackDataB, tmpStackData;
	ivec2 uv0, uv1, uv2, uv3, uv4, uv5, uv6, uv7;

	float d;
	float t = INFINITY;
        float stackptr = 0.0;
	float id = 0.0;
	float tu, tv;
	float triangleID = 0.0;
	float triangleU = 0.0;
	float triangleV = 0.0;
	float triangleW = 0.0;

	bool skip = false;
	bool triangleLookupNeeded = false;
	bool isRayExiting = false;
	intersec.isModel = false;

	
	for (int i = 0; i < N_SPHERES; i++)
        {
		d = SphereIntersect( spheres[i].radius, spheres[i].position, r );
		if (d < t)
		{
			t = d;
			intersec.normal = (r.origin + r.direction * t) - spheres[i].position;
			intersec.emission = spheres[i].emission;
			intersec.color = spheres[i].color;
			intersec.type = spheres[i].type;
			intersec.isModel = false;
		}
        }

	/* // this long cylinder helps to show the direction that points to the sun in this particular HDR image
	d = OpenCylinderIntersect( openCylinders[0].p0, openCylinders[0].p1, openCylinders[0].radius, r, normal );
	if (d < t)
	{
		t = d;
		intersec.normal = normalize(normal);
		intersec.emission = openCylinders[0].emission;
		intersec.color = openCylinders[0].color;
		intersec.type = openCylinders[0].type;
	} */
	
	
	for (int i = 0; i < N_BOXES; i++)
        {
		d = BoxIntersect( boxes[i].minCorner, boxes[i].maxCorner, r, normal, isRayExiting );
		if (d < t)
		{
			t = d;
			intersec.normal = normalize(normal);
			intersec.emission = boxes[i].emission;
			intersec.color = boxes[i].color;
			intersec.type = boxes[i].type;
			intersec.isModel = false;
			finalIsRayExiting = isRayExiting;
		}
	}
	

	currentBoxNode = GetBoxNode(stackptr);
	currentStackData = vec2(stackptr, BoundingBoxIntersect(currentBoxNode.data0.yzw, currentBoxNode.data1.yzw, r.origin, inverseDir));
	stackLevels[0] = currentStackData;
	skip = (currentStackData.y < t);

	while (true)
        {
		if (!skip) 
                {
                        // decrease pointer by 1 (0.0 is root level, 27.0 is maximum depth)
                        if (--stackptr < 0.0) // went past the root level, terminate loop
                                break;

                        currentStackData = stackLevels[int(stackptr)];
			
			if (currentStackData.y >= t)
				continue;
			
			currentBoxNode = GetBoxNode(currentStackData.x);
                }
		skip = false; // reset skip
		

		if (currentBoxNode.data0.x < 0.0) // < 0.0 signifies an inner node
		{
			nodeA = GetBoxNode(currentStackData.x + 1.0);
			nodeB = GetBoxNode(currentBoxNode.data1.x);
			stackDataA = vec2(currentStackData.x + 1.0, BoundingBoxIntersect(nodeA.data0.yzw, nodeA.data1.yzw, r.origin, inverseDir));
			stackDataB = vec2(currentBoxNode.data1.x, BoundingBoxIntersect(nodeB.data0.yzw, nodeB.data1.yzw, r.origin, inverseDir));
			
			// first sort the branch node data so that 'a' is the smallest
			if (stackDataB.y < stackDataA.y)
			{
				tmpStackData = stackDataB;
				stackDataB = stackDataA;
				stackDataA = tmpStackData;

				tmpNode = nodeB;
				nodeB = nodeA;
				nodeA = tmpNode;
			} // branch 'b' now has the larger rayT value of 'a' and 'b'

			if (stackDataB.y < t) // see if branch 'b' (the larger rayT) needs to be processed
			{
				currentStackData = stackDataB;
				currentBoxNode = nodeB;
				skip = true; // this will prevent the stackptr from decreasing by 1
			}
			if (stackDataA.y < t) // see if branch 'a' (the smaller rayT) needs to be processed 
			{
				if (skip) // if larger branch 'b' needed to be processed also,
					stackLevels[int(stackptr++)] = stackDataB; // cue larger branch 'b' for future round
							// also, increase pointer by 1
				
				currentStackData = stackDataA;
				currentBoxNode = nodeA;
				skip = true; // this will prevent the stackptr from decreasing by 1
			}

			continue;
		} // end if (currentBoxNode.data0.x < 0.0) // inner node


		// else this is a leaf

		// each triangle's data is encoded in 8 rgba(or xyzw) texture slots
		id = 8.0 * currentBoxNode.data0.x;

		uv0 = ivec2( mod(id + 0.0, 2048.0), (id + 0.0) * INV_TEXTURE_WIDTH );
		uv1 = ivec2( mod(id + 1.0, 2048.0), (id + 1.0) * INV_TEXTURE_WIDTH );
		uv2 = ivec2( mod(id + 2.0, 2048.0), (id + 2.0) * INV_TEXTURE_WIDTH );
		
		vd0 = texelFetch(tTriangleTexture, uv0, 0);
		vd1 = texelFetch(tTriangleTexture, uv1, 0);
		vd2 = texelFetch(tTriangleTexture, uv2, 0);

		d = BVH_TriangleIntersect( vec3(vd0.xyz), vec3(vd0.w, vd1.xy), vec3(vd1.zw, vd2.x), r, tu, tv );

		if (d < t)
		{
			t = d;
			triangleID = id;
			triangleU = tu;
			triangleV = tv;
			triangleLookupNeeded = true;
		}
	      
        } // end while (true)
	

        if (triangleLookupNeeded)
	{
		//uv0 = ivec2( mod(triangleID + 0.0, 2048.0), (triangleID + 0.0) * INV_TEXTURE_WIDTH );
		//uv1 = ivec2( mod(triangleID + 1.0, 2048.0), (triangleID + 1.0) * INV_TEXTURE_WIDTH );
		uv2 = ivec2( mod(triangleID + 2.0, 2048.0), (triangleID + 2.0) * INV_TEXTURE_WIDTH );
		uv3 = ivec2( mod(triangleID + 3.0, 2048.0), (triangleID + 3.0) * INV_TEXTURE_WIDTH );
		uv4 = ivec2( mod(triangleID + 4.0, 2048.0), (triangleID + 4.0) * INV_TEXTURE_WIDTH );
		uv5 = ivec2( mod(triangleID + 5.0, 2048.0), (triangleID + 5.0) * INV_TEXTURE_WIDTH );
		//uv6 = ivec2( mod(triangleID + 6.0, 2048.0), (triangleID + 6.0) * INV_TEXTURE_WIDTH );
		//uv7 = ivec2( mod(triangleID + 7.0, 2048.0), (triangleID + 7.0) * INV_TEXTURE_WIDTH );
		
		//vd0 = texelFetch(tTriangleTexture, uv0, 0);
		//vd1 = texelFetch(tTriangleTexture, uv1, 0);
		vd2 = texelFetch(tTriangleTexture, uv2, 0);
		vd3 = texelFetch(tTriangleTexture, uv3, 0);
		vd4 = texelFetch(tTriangleTexture, uv4, 0);
		vd5 = texelFetch(tTriangleTexture, uv5, 0);
		//vd6 = texelFetch(tTriangleTexture, uv6, 0);
		//vd7 = texelFetch(tTriangleTexture, uv7, 0);

	
		// face normal for flat-shaded polygon look
		//intersec.normal = normalize( cross(vec3(vd0.w, vd1.xy) - vec3(vd0.xyz), vec3(vd1.zw, vd2.x) - vec3(vd0.xyz)) );
		
		// interpolated normal using triangle intersection's uv's
		triangleW = 1.0 - triangleU - triangleV;
		intersec.normal = normalize(triangleW * vec3(vd2.yzw) + triangleU * vec3(vd3.xyz) + triangleV * vec3(vd3.w, vd4.xy));
		intersec.emission = vec3(0);
		intersec.color = uMaterialColor;//vd6.yzw;
		intersec.uv = triangleW * vec2(vd4.zw) + triangleU * vec2(vd5.xy) + triangleV * vec2(vd5.zw); 
		intersec.type = int(uMaterialType);//int(vd6.x);
		//intersec.albedoTextureID = -1;//int(vd7.x);
		intersec.isModel = true;
	}

	return t;

} // end float SceneIntersect( Ray r, inout Intersection intersec )



vec3 Get_HDR_Color(Ray r)
{
	vec2 sampleUV;
	//sampleUV.x = atan(r.direction.x, -r.direction.z) * ONE_OVER_TWO_PI + 0.5;
	//sampleUV.y = asin(clamp(r.direction.y, -1.0, 1.0)) * ONE_OVER_PI + 0.5;
	sampleUV.x = (1.0 + atan(r.direction.x, -r.direction.z) * ONE_OVER_PI) * 0.5;
  	sampleUV.y = acos(-r.direction.y) * ONE_OVER_PI;
	vec4 texData = texture( tHDRTexture, sampleUV );
	vec3 texColor = vec3(RGBEToLinear(texData));

	return texColor * uHDRI_Exposure;
}


//-----------------------------------------------------------------------
vec3 CalculateRadiance(Ray r)
//-----------------------------------------------------------------------
{
        Intersection intersec;

	vec3 accumCol = vec3(0);
        vec3 mask = vec3(1);
	vec3 checkCol0 = vec3(1);
	vec3 checkCol1 = vec3(0.5);
        vec3 tdir;
	vec3 x, n, nl;
        
	float t;
        float nc, nt, ratioIoR, Re, Tr;
	float P, RP, TP;
	float weight;
	float thickness = 0.1;
	float roughness = 0.0;

	int diffuseCount = 0;

	bool bounceIsSpecular = true;
	bool sampleLight = false;
	bool isRayExiting = false;
	
	
	for (int bounces = 0; bounces < 5; bounces++)
	{

		t = SceneIntersect(r, intersec, isRayExiting);
		roughness = intersec.isModel ? uRoughness : roughness;
		
		
		if (t == INFINITY)
		{	
                        vec3 environmentCol = Get_HDR_Color(r);

			// looking directly at sky
			if (bounces == 0)
			{
				accumCol = environmentCol;
				break;
			}

			

			// sun light source location in HDRI image is sampled by a diffuse surface
			// mask has already been down-weighted in this case
			if (sampleLight)
			{
				accumCol = mask * environmentCol;
				break;
			}

			// sky image seen in a reflection or refraction
			// if (bounceIsSpecular)
			// {
			// 	// try to get rid of fireflies on rougher surfaces
			// 	if (dot(r.direction, SUN_DIRECTION) > 0.98)
			// 		environmentCol = vec3(1);
			// 		//environmentCol = mix( vec3(1), environmentCol, clamp(pow(roughness, 0.0), 0.0, 1.0) );
				
			// 	accumCol = mask * environmentCol;
			// 	break;
			// }

			// random diffuse bounce hits sky
			if ( !bounceIsSpecular )
			{
				weight = dot(r.direction, SUN_DIRECTION) < 0.98 ? 1.0 : 0.0;
				accumCol = mask * environmentCol * weight;

				if (bounces == 3)
					accumCol = mask * environmentCol * weight * 2.0; // ground beneath glass table

				break;
			}
			
			if (bounceIsSpecular)
			{
				if (dot(r.direction, SUN_DIRECTION) > 0.8)
					environmentCol = mix(vec3(1), environmentCol, clamp(pow(1.0-roughness, 20.0), 0.0, 1.0));
				accumCol = mask * environmentCol;
				break;
			}
					
			// reached the HDRI sky light, so we can exit
			break;
		} // end if (t == INFINITY)


		// if we get here and sampleLight is still true, shadow ray failed to find a light source
		if (sampleLight) 
			break;
		
		
		// useful data 
		n = normalize(intersec.normal);
                nl = dot(n, r.direction) < 0.0 ? n : normalize(-n);
		x = r.origin + r.direction * t;

		    
                if (intersec.type == DIFF || intersec.type == CHECK) // Ideal DIFFUSE reflection
                {
			diffuseCount++;

			if( intersec.type == CHECK )
			{
				float q = clamp( mod( dot( floor(x.xz * 0.04), vec2(1.0) ), 2.0 ) , 0.0, 1.0 );
				intersec.color = checkCol0 * q + checkCol1 * (1.0 - q);	
			}

			mask *= intersec.color;

			bounceIsSpecular = false;
                        
			if (diffuseCount == 1 && rand() < 0.5)
			{
				// choose random Diffuse sample vector
				r = Ray( x, randomCosWeightedDirectionInHemisphere(nl) );
				r.origin += nl * uEPS_intersect;
				continue;
			}

			r = Ray( x, SUN_DIRECTION );
			r.direction = randomDirectionInSpecularLobe(r.direction, 0.01);
			r.origin += nl * uEPS_intersect;
			weight = max(0.0, dot(r.direction, nl)) * 0.00002; // down-weight directSunLight contribution
			mask *= weight;

			sampleLight = true;
			continue;
                }
		
                if (intersec.type == SPEC)  // Ideal SPECULAR reflection
                {
			mask *= intersec.color;
			
			r = Ray( x, reflect(r.direction, nl) );
			r.direction = randomDirectionInSpecularLobe(r.direction, roughness);
			
			r.origin += nl * uEPS_intersect;
                        continue;
                }

                if (intersec.type == REFR)  // Ideal dielectric REFRACTION
		{
			nc = 1.0; // IOR of Air
			nt = 1.5; // IOR of common Glass
			Re = calcFresnelReflectance(r.direction, n, nc, nt, ratioIoR);
			Tr = 1.0 - Re;
			P  = 0.25 + (0.5 * Re);
                	RP = Re / P;
                	TP = Tr / (1.0 - P);

			if (rand() < P)
			{
				mask *= RP;
				r = Ray( x, reflect(r.direction, nl) );
				r.direction = randomDirectionInSpecularLobe(r.direction, roughness);
				r.origin += nl * uEPS_intersect;
				continue;
			}
			
			// transmit ray through surface
			mask *= TP;

			// is ray leaving a solid object from the inside? 
			// If so, attenuate ray color with object color by how far ray has travelled through the medium
			if (isRayExiting || (distance(n, nl) > 0.1))
			{
				isRayExiting = false;
				mask *= exp(log(intersec.color) * thickness * t);
			}
			else 
				mask *= intersec.color;
			
			tdir = refract(r.direction, nl, ratioIoR);
			r = Ray(x, tdir);
			r.direction = randomDirectionInSpecularLobe(r.direction, roughness * roughness);
			r.origin -= nl * uEPS_intersect;

			continue;
			
		} // end if (intersec.type == REFR)
		
		if (intersec.type == COAT)  // Diffuse object underneath with ClearCoat on top (like car, or shiny pool ball)
		{
			nc = 1.0; // IOR of Air
			nt = 1.5; // IOR of Clear Coat
			Re = calcFresnelReflectance(r.direction, n, nc, nt, ratioIoR);
			Tr = 1.0 - Re;
			P  = 0.25 + (0.5 * Re);
                	RP = Re / P;
                	TP = Tr / (1.0 - P);

			if (rand() < P)
			{
				mask *= RP;
				r = Ray( x, reflect(r.direction, nl) );
				r.direction = randomDirectionInSpecularLobe(r.direction, roughness);
				r.origin += nl * uEPS_intersect;
				continue;
			}

			diffuseCount++;

			mask *= intersec.color;
			mask *= TP;
			
			bounceIsSpecular = false;

			if (diffuseCount == 1 && rand() < 0.5)
			{
				// choose random Diffuse sample vector
				r = Ray( x, randomCosWeightedDirectionInHemisphere(nl) );
				r.origin += nl * uEPS_intersect;
				continue;
			}

			r = Ray( x, SUN_DIRECTION );
			r.direction = randomDirectionInSpecularLobe(r.direction, 0.01);
			r.origin += nl * uEPS_intersect;
			weight = max(0.0, dot(r.direction, nl)) * 0.00002; // down-weight directSunLight contribution
			mask *= weight;

			sampleLight = true;
			continue;
			
		} //end if (intersec.type == COAT)
		
		
	} // end for (int bounces = 0; bounces < 5; bounces++)
	

	return max(vec3(0), accumCol);

} // end vec3 CalculateRadiance(Ray r)


//-----------------------------------------------------------------------
void SetupScene(void)
//-----------------------------------------------------------------------
{
	vec3 z  = vec3(0);          
	
	spheres[0] = Sphere(  4000.0, vec3(0, -4000, 0),  z, vec3(0.4,0.4,0.4), CHECK);//Checkered Floor
	spheres[1] = Sphere(     6.0, vec3(55, 36, -45),  z,         vec3(0.9),  SPEC);//small mirror ball
	spheres[2] = Sphere(     6.0, vec3(55, 24, -45),  z, vec3(0.5,1.0,1.0),  REFR);//small glass ball
	spheres[3] = Sphere(     6.0, vec3(60, 24, -30),  z,         vec3(1.0),  COAT);//small plastic ball
	
	// this long red metal cylinder points to the sun in an HDR image.  Unfortunately for a random image, the SUN_DIRECTION
	// vector pointing directly to the sun has to be found by trial and error. To see this helper, uncomment cylinder intersection code above 
	openCylinders[0] = OpenCylinder(vec3(0, 0, 0), SUN_DIRECTION * 100000.0, 10.0, z, vec3(1, 0, 0), SPEC);

	boxes[0] = Box( vec3(-20.0,11.0,-110.0), vec3(70.0,18.0,-20.0), z, vec3(0.2,0.9,0.7), REFR);//Glass Box
	boxes[1] = Box( vec3(-14.0,13.0,-104.0), vec3(64.0,16.0,-26.0), z, vec3(0),           DIFF);//Inner Box
}


#include <pathtracing_main>
