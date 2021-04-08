precision highp float;
precision highp int;
precision highp sampler2D;

uniform mat4 uShortBoxInvMatrix;
uniform mat4 uTallBoxInvMatrix;

#include <pathtracing_uniforms_and_defines>

#define N_QUADS 6
#define N_BOXES 2


struct Ray {
    vec3 origin;
    vec3 direction;
};

struct Quad {
    vec3 normal;
    vec3 v0;
    vec3 v1;
    vec3 v2;
    vec3 v3;
    vec3 emission;
    vec3 color;
    int type;
};


struct Box { 
    vec3 minCorner;
    vec3 maxCorner;
    vec3 emission;
    vec3 color;
    int type;
};

struct Intersection {
    vec3 normal;    //相交点的法线
    vec3 emission;  //自发光颜色
    vec3 color;     //基础色
    int type;
};

Quad quads[N_QUADS];
Box boxes[N_BOXES];

// #include <pathtracing_random_functions>

// globals used in rand() function
vec4 randVec4 = vec4(0); // samples and holds the RGBA blueNoise texture value for this pixel
float randNumber = 0.0; // the final randomly generated number (range: 0.0 to 1.0)
float counter = -1.0; // will get incremented by 1 on each call to rand()
int channel = 0; // the final selected color channel to use for rand() calc (range: 0 to 3, corresponds to R,G,B, or A)

float rand()
{
	counter++; // increment counter by 1 on every call to rand()
	// cycles through channels, if modulus is 1.0, channel will always be 0 (the R color channel)
	channel = int(mod(counter, 4.0)); 
	// but if modulus was 4.0, channel will cycle through all available channels: 0,1,2,3,0,1,2,3, and so on...
	randNumber = randVec4[channel]; // get value stored in channel 0:R, 1:G, 2:B, or 3:A
	return fract(randNumber); // we're only interested in randNumber's fractional value between 0.0 (inclusive) and 1.0 (non-inclusive)
}
// from iq https://www.shadertoy.com/view/4tXyWN
// global seed used in rng() function
uvec2 seed;
//这里用来生成噪声
float rng()
{
	seed += uvec2(1);
    	uvec2 q = 1103515245U * ( (seed >> 1U) ^ (seed.yx) );
    	uint  n = 1103515245U * ( (q.x) ^ (q.y >> 3U) );
	return float(n) * (1.0 / float(0xffffffffU));
}
vec3 randomSphereDirection()
{
    	float up = rand() * 2.0 - 1.0; // range: -1 to +1
	float over = sqrt( max(0.0, 1.0 - up * up) );
	float around = rand() * TWO_PI;
	return normalize(vec3(cos(around) * over, up, sin(around) * over));	
}
vec3 randomDirectionInHemisphere(vec3 nl)
{
	float r = rand(); // uniform distribution in hemisphere
	float phi = rand() * TWO_PI;
	float x = r * cos(phi);
	float y = r * sin(phi);
	float z = sqrt(1.0 - x*x - y*y);
	
	vec3 U = normalize( cross(vec3(0.7071067811865475, 0.7071067811865475, 0), nl ) );
	vec3 V = cross(nl, U);
	return normalize(x * U + y * V + z * nl);
}
vec3 randomCosWeightedDirectionInHemisphere(vec3 nl)
{
	float r = sqrt(rand()); // cos-weighted distribution in hemisphere
	float phi = rand() * TWO_PI;
	float x = r * cos(phi);
	float y = r * sin(phi);
	float z = sqrt(1.0 - x*x - y*y);
	
	vec3 U = normalize( cross(vec3(0.7071067811865475, 0.7071067811865475, 0), nl ) );
	vec3 V = cross(nl, U);
	return normalize(x * U + y * V + z * nl);
}

vec3 randomDirectionInSpecularLobe(vec3 reflectionDir, float roughness)
{
	roughness = clamp(roughness, 0.0, 1.0);
	float exponent = mix(7.0, 0.0, sqrt(roughness));
	float cosTheta = pow(rand(), 1.0 / (exp(exponent) + 1.0));
	float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
	float phi = rand() * TWO_PI;
	
	vec3 U = normalize( cross(vec3(0.7071067811865475, 0.7071067811865475, 0), reflectionDir ) );
	vec3 V = cross(reflectionDir, U);
	return normalize(mix(reflectionDir, (U * cos(phi) * sinTheta + V * sin(phi) * sinTheta + reflectionDir * cosTheta), roughness));
}


// #include <pathtracing_quad_intersect>

//计算光线与三角形的相交距离，如果未相交，则返回INFINITY
float TriangleIntersect( vec3 v0, vec3 v1, vec3 v2, Ray r, bool isDoubleSided )
{
	vec3 edge1 = v1 - v0;
	vec3 edge2 = v2 - v0;
	vec3 pvec = cross(r.direction, edge2);
	float det = 1.0 / dot(edge1, pvec);
	if ( !isDoubleSided && det < 0.0 ) 
		return INFINITY;
	vec3 tvec = r.origin - v0;
	float u = dot(tvec, pvec) * det;
	vec3 qvec = cross(tvec, edge1);
	float v = dot(r.direction, qvec) * det;
	float t = dot(edge2, qvec) * det;
	return (u < 0.0 || u > 1.0 || v < 0.0 || u + v > 1.0 || t <= 0.0) ? INFINITY : t;
}
//----------------------------------------------------------------------------------
float QuadIntersect( vec3 v0, vec3 v1, vec3 v2, vec3 v3, Ray r, bool isDoubleSided )
//----------------------------------------------------------------------------------
{
	return min(TriangleIntersect(v0, v1, v2, r, isDoubleSided), TriangleIntersect(v0, v2, v3, r, isDoubleSided));
}

// #include <pathtracing_box_intersect>

//-------------------------------------------------------------------------------------------------------
//计算光线与box的相交
float BoxIntersect( vec3 minCorner, vec3 maxCorner, inout Ray r, out vec3 normal, out bool isRayExiting )
//-------------------------------------------------------------------------------------------------------
{
	//r.direction = normalize(r.direction);
	vec3 invDir = 1.0 / r.direction;
	vec3 near = (minCorner - r.origin) * invDir;
	vec3 far  = (maxCorner - r.origin) * invDir;
	
	vec3 tmin = min(near, far);
	vec3 tmax = max(near, far);
	
	float t0 = max( max(tmin.x, tmin.y), tmin.z);
	float t1 = min( min(tmax.x, tmax.y), tmax.z);
	
	if (t0 > t1) return INFINITY;
	if (t0 > 0.0) // if we are outside the box
	{
		normal = -sign(r.direction) * step(tmin.yzx, tmin) * step(tmin.zxy, tmin);
		isRayExiting = false;
		return t0;	
	}
	if (t1 > 0.0) // if we are inside the box
	{
		normal = -sign(r.direction) * step(tmax, tmax.yzx) * step(tmax, tmax.zxy);
		isRayExiting = true;
		return t1;
	}
	return INFINITY;
}

#include <pathtracing_sample_quad_light>


//-----------------------------------------------------------------------
//场景相交测试，找到最近相交的那个，即 参数 d 最小
float SceneIntersect( Ray r, inout Intersection intersec )
//-----------------------------------------------------------------------
{
	vec3 normal;
    float d;
	float t = INFINITY;
	bool isRayExiting = false;  //光线是否停止
		
    //计算光线与房间和顶部面光源的相交
	for (int i = 0; i < N_QUADS; i++){
        //循环遍历每个四边形面，计算是否相交
		d = QuadIntersect( quads[i].v0, quads[i].v1, quads[i].v2, quads[i].v3, r, false );
        //如果相交了
		if (d < t){
			t = d;
			intersec.normal = normalize(quads[i].normal);
			intersec.emission = quads[i].emission;
			intersec.color = quads[i].color;
			intersec.type = quads[i].type;
		}
    }
	
	
	// TALL MIRROR BOX
    //执行 高的镜面盒子的路径追踪
	Ray rObj;
	// transform ray into Tall Box's object space
	rObj.origin = vec3( uTallBoxInvMatrix * vec4(r.origin, 1.0) );
	rObj.direction = vec3( uTallBoxInvMatrix * vec4(r.direction, 0.0) );
    //这里计算与镜面立方体盒子的相交
	d = BoxIntersect( boxes[0].minCorner, boxes[0].maxCorner, rObj, normal, isRayExiting );
	
    //如果相交
	if (d < t)
	{	
		t = d;
		
		// transfom normal back into world space
		normal = normalize(normal);
        //这里将法线转换到世界坐标系下
		intersec.normal = normalize(transpose(mat3(uTallBoxInvMatrix)) * normal);
		intersec.emission = boxes[0].emission;
		intersec.color = boxes[0].color;
		intersec.type = boxes[0].type;
	}
	
	
	// SHORT DIFFUSE WHITE BOX 执行矮的白色立方体的路径追踪
	// transform ray into Short Box's object space
	rObj.origin = vec3( uShortBoxInvMatrix * vec4(r.origin, 1.0) );
	rObj.direction = vec3( uShortBoxInvMatrix * vec4(r.direction, 0.0) );
	d = BoxIntersect( boxes[1].minCorner, boxes[1].maxCorner, rObj, normal, isRayExiting );
	
    //如果相交
	if (d < t)
	{	
		t = d;
		
		// transfom normal back into world space
		normal = normalize(normal);
		intersec.normal = normalize(transpose(mat3(uShortBoxInvMatrix)) * normal);
		intersec.emission = boxes[1].emission;
		intersec.color = boxes[1].color;
		intersec.type = boxes[1].type;
	}
	
	
	return t;
}


//-----------------------------------------------------------------------
//射线计算辐照度 (光线)
vec3 CalculateRadiance(Ray r)
//-----------------------------------------------------------------------
{
        Intersection intersec;
        Quad light = quads[5];

	vec3 accumCol = vec3(0);
    vec3 mask = vec3(1);
    vec3 n, nl, x;
	vec3 dirToLight;
        
	float t;
	float weight, p;
	
	int diffuseCount = 0;

	bool bounceIsSpecular = true;
	bool sampleLight = false;
	bool createCausticRay = false;


    //射线这里弹射5次
	for (int bounces = 0; bounces < 5; bounces++){
        //场景相交测试
		t = SceneIntersect(r, intersec);
		
        //如果与场景中的对象均没有相交
		if (t == INFINITY)
			break;
		
        //如果相交的对象是灯光，则结束此次弹射
		if (intersec.type == LIGHT){	
			if (bounceIsSpecular || sampleLight || createCausticRay)
				accumCol = mask * intersec.emission;

			// if (createCausticRay)
			// 	accumCol = mask * intersec.emission * max(0.0, dot(-r.direction, normalize(intersec.normal)));

			// reached a light source, so we can exit
			break;
		}
		
		// if we get here and sampleLight is still true, shadow ray failed to find a light source
        //如果光线经过几次弹射之后依然无法击中光源，则结束此次弹射
		if (sampleLight) 
			break;
		
		// useful data  法线归一化
		n = normalize(intersec.normal);
        nl = dot(n, r.direction) < 0.0 ? normalize(n) : normalize(-n);
        //这里计算相交的那个点
		x = r.origin + r.direction * t;
		
		//这里材质如果是漫反射
        if (intersec.type == DIFF) // Ideal DIFFUSE reflection
        {
			diffuseCount++;

			mask *= intersec.color;

			if (createCausticRay)
				break;

			// create caustic ray
            //第一次漫反射
            if (diffuseCount == 1 && rand() < 0.25 && uSampleCounter > 20.0)
            {
				createCausticRay = true;

				vec3 randVec = vec3(rand() * 2.0 - 1.0, rand() * 2.0 - 1.0, rand() * 2.0 - 1.0);
				vec3 offset = vec3(randVec.x * 82.0, randVec.y * 170.0, randVec.z * 80.0);
				vec3 target = vec3(180.0 + offset.x, 170.0 + offset.y, -350.0 + offset.z);
				r = Ray( x, normalize(target - x) );
				r.origin += nl * uEPS_intersect;
				
				weight = max(0.0, dot(nl, r.direction));
				mask *= weight;

				continue;
			}

			bounceIsSpecular = false;

			if (diffuseCount == 1 && rand() < 0.5)
			{	
				// choose random Diffuse sample vector
				r = Ray( x, randomCosWeightedDirectionInHemisphere(nl) );
				r.origin += nl * uEPS_intersect;
				continue;
			}
			
			dirToLight = sampleQuadLight(x, nl, light, weight);
			mask *= weight;

			r = Ray( x, dirToLight );
			r.origin += nl * uEPS_intersect;
			sampleLight = true;
			continue;
                        
        } // end if (intersec.type == DIFF)
		
        //镜面反射
        if (intersec.type == SPEC)  // Ideal SPECULAR reflection
		{
			mask *= intersec.color;

			r = Ray( x, reflect(r.direction, nl) );
			r.origin += nl * uEPS_intersect;

			continue;
		}
		
	} // end for (int bounces = 0; bounces < 5; bounces++)
	

	return max(vec3(0), accumCol);

}


//-----------------------------------------------------------------------
void SetupScene(void)
//-----------------------------------------------------------------------
{
	vec3 z  = vec3(0);// No color value, Black        
	vec3 L1 = vec3(1.0, 0.7, 0.38) * 30.0;// Bright Yellowish light
	

	quads[0] = Quad(
        //法线
        vec3(0.0, 0.0, 1.0),
        //面的四个点
        vec3(0.0,   0.0,-559.2),
        vec3(549.6,   0.0,-559.2),
        vec3(549.6, 548.8,-559.2),
        vec3(0.0, 548.8,-559.2), 
        //自发光颜色
        z,
        //材质本身颜色
        vec3(1),
        //对象类型
        DIFF
    );// Back Wall
	quads[1] = Quad( vec3( 1.0, 0.0, 0.0), vec3(  0.0,   0.0,   0.0), vec3(  0.0,   0.0,-559.2), vec3(  0.0, 548.8,-559.2), vec3(  0.0, 548.8,   0.0),  z, vec3(0.7, 0.12,0.05),  DIFF);// Left Wall Red
	quads[2] = Quad( vec3(-1.0, 0.0, 0.0), vec3(549.6,   0.0,-559.2), vec3(549.6,   0.0,   0.0), vec3(549.6, 548.8,   0.0), vec3(549.6, 548.8,-559.2),  z, vec3(0.2, 0.4, 0.36),  DIFF);// Right Wall Green
	quads[3] = Quad( vec3( 0.0,-1.0, 0.0), vec3(  0.0, 548.8,-559.2), vec3(549.6, 548.8,-559.2), vec3(549.6, 548.8,   0.0), vec3(  0.0, 548.8,   0.0),  z, vec3(1),  DIFF);// Ceiling
	quads[4] = Quad( vec3( 0.0, 1.0, 0.0), vec3(  0.0,   0.0,   0.0), vec3(549.6,   0.0,   0.0), vec3(549.6,   0.0,-559.2), vec3(  0.0,   0.0,-559.2),  z, vec3(1),  DIFF);// Floor


	quads[5] = Quad( vec3( 0.0,-1.0, 0.0), vec3(213.0, 548.0,-332.0), vec3(343.0, 548.0,-332.0), vec3(343.0, 548.0,-227.0), vec3(213.0, 548.0,-227.0), L1,       z, LIGHT);// Area Light Rectangle in ceiling
    
    //SPEC
	boxes[0]  = Box( vec3(-82.0,-170.0, -80.0), vec3(82.0,170.0, 80.0), z, vec3(1), DIFF);// Tall Mirror Box Left
	boxes[1]  = Box( vec3(-86.0, -85.0, -80.0), vec3(86.0, 85.0, 80.0), z, vec3(1), DIFF);// Short Diffuse Box Right
}


// #include <pathtracing_main>

// tentFilter from Peter Shirley's 'Realistic Ray Tracing (2nd Edition)' book, pg. 60		
float tentFilter(float x)
{
	return (x < 0.5) ? sqrt(2.0 * x) - 1.0 : 1.0 - sqrt(2.0 - (2.0 * x));
}

void main( void )
{
	//相机的右方向
	vec3 camRight   = vec3( uCameraMatrix[0][0],  uCameraMatrix[0][1],  uCameraMatrix[0][2]);
    //相机的上方向
	vec3 camUp      = vec3( uCameraMatrix[1][0],  uCameraMatrix[1][1],  uCameraMatrix[1][2]);
    //相机的视线方向
	vec3 camForward = vec3(-uCameraMatrix[2][0], -uCameraMatrix[2][1], -uCameraMatrix[2][2]);
	// the following is not needed - three.js has a built-in uniform named cameraPosition
	//vec3 camPos   = vec3( uCameraMatrix[3][0],  uCameraMatrix[3][1],  uCameraMatrix[3][2]);
	
	// calculate unique seed for rng() function
    //计算一个生成随机数的种子
	seed = uvec2(uFrameCounter, uFrameCounter + 1.0) * uvec2(gl_FragCoord); // old way of generating random numbers
    //这里使用噪声图来模拟随机数
	randVec4 = texture(tBlueNoiseTexture, (gl_FragCoord.xy + (uRandomVec2 * 255.0)) / 255.0 ); // new way of rand()

    
	vec2 pixelOffset = vec2( tentFilter(rng()), tentFilter(rng()) );
	// we must map pixelPos into the range -1.0 to +1.0
    //将像素左边转换到[-1, 1]范围内
	vec2 pixelPos = ((gl_FragCoord.xy + pixelOffset) / uResolution) * 2.0 - 1.0;
	vec3 rayDir = normalize( pixelPos.x * camRight * uULen + pixelPos.y * camUp * uVLen + camForward );
	
	// depth of field
	vec3 focalPoint = uFocusDistance * rayDir;
	float randomAngle = rand() * TWO_PI; // pick random point on aperture
	float randomRadius = rand() * uApertureSize;
    //基于该像素计算的随机点
	vec3  randomAperturePos = ( cos(randomAngle) * camRight + sin(randomAngle) * camUp ) * sqrt(randomRadius);
	// point on aperture to focal point
    //计算当前射线的方向
	vec3 finalRayDir = normalize(focalPoint - randomAperturePos);
	
    //生成射线
	Ray ray = Ray( cameraPosition + randomAperturePos , finalRayDir );

    //初始化整个场景对象(组成房间的5个面，顶部的面灯光，左侧的反射box，右侧的粗糙盒子)
	SetupScene();
				
	// perform path tracing and get resulting pixel color
    //根据射线计算辐照度颜色
	vec3 pixelColor = CalculateRadiance(ray);
	
	vec3 previousColor = texelFetch(tPreviousTexture, ivec2(gl_FragCoord.xy), 0).rgb;
	if (uFrameCounter == 1.0) // camera just moved after being still
	{
		previousColor = vec3(0); // clear rendering accumulation buffer
	}
	else if (uCameraIsMoving) // camera is currently moving
	{
		previousColor *= 0.5; // motion-blur trail amount (old image)
		pixelColor *= 0.5; // brightness of new image (noisy)
	}
		
	pc_fragColor = vec4( pixelColor + previousColor, 1.0 );
}