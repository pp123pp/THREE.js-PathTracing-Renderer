precision highp float;
precision highp int;
precision highp sampler2D;

uniform vec3 uSunDirection;
uniform float uWaterLevel;

#include <pathtracing_uniforms_and_defines>

uniform sampler2D t_PerlinNoise;

#include <pathtracing_skymodel_defines>


//-----------------------------------------------------------------------

struct Ray { vec3 origin; vec3 direction; };
struct Quad { vec3 v0; vec3 v1; vec3 v2; vec3 v3; vec3 emission; vec3 color; int type; };
struct Box { vec3 minCorner; vec3 maxCorner; vec3 emission; vec3 color; int type; };
struct Intersection { vec3 normal; vec3 emission; vec3 color; vec2 uv; int type; };


#include <pathtracing_random_functions>

#include <pathtracing_calc_fresnel_reflectance>

#include <pathtracing_sphere_intersect>

#include <pathtracing_physical_sky_functions>



// TERRAIN

#define TERRAIN_FAR 100000.0
#define TERRAIN_HEIGHT 2000.0 // terrain amplitude
#define TERRAIN_LIFT  -1900.0 // how much to lift/drop the entire terrain
#define TERRAIN_SAMPLE_SCALE 0.00001

float lookup_Heightmap( in vec3 pos )
{
	vec2 uv = pos.xz;
	uv *= TERRAIN_SAMPLE_SCALE;
	float h = 0.0;
	float mult = 1.0;
	float scaleAccum = mult;

	for (int i = 0; i < 3; i ++)
	{
		h += mult * texture(t_PerlinNoise, uv + 0.5).x;
		mult *= 0.5;
		uv *= 2.0;
	}
	return h * TERRAIN_HEIGHT + TERRAIN_LIFT;	
}

float lookup_Normal( in vec3 pos )
{
	vec2 uv = pos.xz;
	uv *= TERRAIN_SAMPLE_SCALE;
	float h = 0.0;
	float mult = 1.0;
	float scaleAccum = mult;

	for (int i = 0; i < 9; i ++)
	{
		h += mult * texture(t_PerlinNoise, uv + 0.5).x;
		mult *= 0.5;
		uv *= 2.0;
	}
	return h  * TERRAIN_HEIGHT + TERRAIN_LIFT;
}

vec3 terrain_calcNormal( vec3 pos, float t )
{
	vec3 eps = vec3(uEPS_intersect, 0.0, 0.0);
	
	return normalize( vec3( lookup_Normal(pos-eps.xyy) - lookup_Normal(pos+eps.xyy),
			  	eps.x * 2.0,
			  	lookup_Normal(pos-eps.yyx) - lookup_Normal(pos+eps.yyx) ) );
}

float TerrainIntersect( Ray r )
{
	vec3 pos = r.origin;
	vec3 dir = normalize(r.direction);
	float h = 0.0;
	float t = 0.0;
	float epsilon = 1.0;
	
	for(int i = 0; i < 200; i++)
	{
		h = pos.y - lookup_Heightmap(pos);
		if (t > TERRAIN_FAR || h < epsilon) break;
		t += h * 0.6;
		pos += dir * h * 0.6; 
	}
	return (h <= epsilon) ? t : INFINITY;	    
}

bool isLightSourceVisible( vec3 pos, vec3 n, vec3 dirToLight)
{
	dirToLight = normalize(dirToLight);
	float h = 1.0;
	float t = 0.0;
	float terrainHeight = TERRAIN_HEIGHT * 1.5 + TERRAIN_LIFT;
	
	for(int i = 0; i < 100; i++)
	{
		h = pos.y - lookup_Heightmap(pos);
		if ( pos.y > terrainHeight || h < 0.0) break;
		pos += dirToLight * h;
	}

	return h > 0.0;
}

// WATER
/* Credit: some of the following water code is borrowed from https://www.shadertoy.com/view/Ms2SD1 posted by user 'TDM' */

#define WATER_COLOR vec3(0.96, 1.0, 0.98)
#define WATER_SAMPLE_SCALE 0.009 
#define WATER_WAVE_HEIGHT 20.0 // max height of water waves   
#define WATER_FREQ        0.2 // wave density: lower = spread out, higher = close together
#define WATER_CHOPPY      1.9 // smaller beachfront-type waves, they travel in parallel
#define WATER_SPEED       1.7 // how quickly time passes
#define M1  mat2(1.6, 1.2, -1.2, 1.6);

float hash( vec2 p )
{
	float h = dot(p,vec2(127.1,311.7));	
    	return fract(sin(h)*43758.5453123);
}

float noise( in vec2 p )
{
	vec2 i = floor( p );
	vec2 f = fract( p );	
	vec2 u = f*f*(3.0-2.0*f);
	return -1.0+2.0*mix( mix( hash( i + vec2(0.0,0.0) ), 
		     hash( i + vec2(1.0,0.0) ), u.x),
		mix( hash( i + vec2(0.0,1.0) ), 
		     hash( i + vec2(1.0,1.0) ), u.x), u.y);
}

float ocean_octave( vec2 uv, float choppy )
{
	uv += noise(uv);        
	vec2 wv = 1.0 - abs(sin(uv));
	vec2 swv = abs(cos(uv));    
	wv = mix(wv, swv, wv);
	return pow(1.0 - pow(wv.x * wv.y, 0.65), choppy);
}

float getOceanWaterHeight( vec3 p )
{
	float freq = WATER_FREQ;
	float amp = 1.0;
	float choppy = WATER_CHOPPY;
	float sea_time = uTime * WATER_SPEED;
	
	vec2 uv = p.xz * WATER_SAMPLE_SCALE; 
	//uv.x *= 0.75;
	float h, d = 0.0;    
	for(int i = 0; i < 1; i++)
	{        
		d =  ocean_octave((uv + sea_time) * freq, choppy);
		d += ocean_octave((uv - sea_time) * freq, choppy);
		h += d * amp;     
		uv *= M1; 
		freq *= 1.9; 
		amp *= 0.22;
		choppy = mix(choppy, 1.0, 0.2);
	}

	return h * WATER_WAVE_HEIGHT + uWaterLevel;
}

float getOceanWaterHeight_Detail( vec3 p )
{
	float freq = WATER_FREQ;
	float amp = 1.0;
	float choppy = WATER_CHOPPY;
	float sea_time = uTime * WATER_SPEED;
	
	vec2 uv = p.xz * WATER_SAMPLE_SCALE; 
	//uv.x *= 0.75;
	float h, d = 0.0;    
	for(int i = 0; i < 4; i++)
	{        
		d =  ocean_octave((uv + sea_time) * freq, choppy);
		d += ocean_octave((uv - sea_time) * freq, choppy);
		h += d * amp;     
		uv *= M1; 
		freq *= 1.9; 
		amp *= 0.22;
		choppy = mix(choppy, 1.0, 0.2);
	}

	return h * WATER_WAVE_HEIGHT + uWaterLevel;
}


float OceanIntersect( Ray r )
{
	vec3 pos = r.origin;
	vec3 dir = (r.direction);
	float h = 0.0;
	float t = 0.0;
	
	for(int i = 0; i < 200; i++)
	{
		h = abs(pos.y - getOceanWaterHeight(pos));
		if (t > TERRAIN_FAR || h < 1.0) break;
		t += h;
		pos += dir * h; 
	}
	return (h <= 1.0) ? t : INFINITY;
}

vec3 ocean_calcNormal( vec3 pos, float t )
{
	vec3 eps = vec3(1.0, 0.0, 0.0);
	
	return normalize( vec3( getOceanWaterHeight_Detail(pos-eps.xyy) - getOceanWaterHeight_Detail(pos+eps.xyy),
			  	eps.x * 2.0,
			  	getOceanWaterHeight_Detail(pos-eps.yyx) - getOceanWaterHeight_Detail(pos+eps.yyx) ) );
}


//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
float SceneIntersect( Ray r, inout Intersection intersec, bool checkWater )
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
{
	vec3 normal;
        float d, dw;
	float t = INFINITY;
	vec3 hitPos; 

	// Terrain
	d = TerrainIntersect( r );
	if (d < t)
	{
		t = d;
		hitPos = r.origin + r.direction * t;
		intersec.normal = terrain_calcNormal(hitPos, t);
		intersec.emission = vec3(0);
		intersec.color = vec3(0);
		intersec.type = TERRAIN;
	}
	
	if (!checkWater)
		return t;
        
        d = OceanIntersect( r );    
	
	if (d < t)
	{
                t = d;
                hitPos = r.origin + r.direction * d;
		intersec.normal = ocean_calcNormal(hitPos, t);
		intersec.emission = vec3(0);
		intersec.color = vec3(0.7,0.8,0.9);
		intersec.type = REFR;
	}
		
	return t;
}


//-----------------------------------------------------------------------
vec3 CalculateRadiance(Ray r, vec3 sunDirection)
//-----------------------------------------------------------------------
{
	Intersection intersec;
	Ray firstRay;
	Ray cameraRay = r;

	vec3 randVec = vec3(rand() * 2.0 - 1.0, rand() * 2.0 - 1.0, rand() * 2.0 - 1.0);
	vec3 initialSkyColor = Get_Sky_Color(r, normalize(sunDirection));
	
	//Ray skyRay = Ray( r.origin * vec3(0.05), normalize(vec3(r.direction.x, abs(r.direction.y), r.direction.z)) );
	//float dc = SphereIntersect( 10000.0, vec3(skyRay.origin.x, -9800, skyRay.origin.z) + vec3(rand() * 2.0), skyRay );
	//vec3 skyPos = skyRay.origin + skyRay.direction * dc;
	//vec4 cld = render_clouds(skyRay, skyPos, sunDirection);
	
	vec3 accumCol = vec3(0);
        vec3 mask = vec3(1);
	vec3 firstMask = vec3(1);
	vec3 n, nl, x;
	vec3 firstX = vec3(0);
	vec3 tdir;
	
	float nc, nt, ratioIoR, Re, Tr;
	float t = INFINITY;
	float cameraUnderWaterValue = r.origin.y < getOceanWaterHeight(r.origin) ? 1.0 : 0.0;
	float thickness = 0.01;

	bool checkWater = true;
	bool skyHit = false;
	bool firstTypeWasREFR = false;
	bool reflectionTime = false;
	bool rayEnteredWater = false;

	
        for (int bounces = 0; bounces < 3; bounces++)
	{

		t = SceneIntersect(r, intersec, checkWater);
		checkWater = false;
		
		if (t == INFINITY)
		{
			if (bounces == 0) // ray hits sky first	
			{
				skyHit = true;
				//firstX = skyPos;
				accumCol = initialSkyColor;
				break; // exit early	
			}

			if (firstTypeWasREFR)
			{
				if (!reflectionTime) 
				{
					accumCol = mask * Get_Sky_Color(r, normalize(sunDirection));
					
					// start back at the refractive surface, but this time follow reflective branch
					r = firstRay;
					mask = firstMask;
					// set/reset variables
					reflectionTime = true;
					// continue with the reflection ray
					continue;
				}

				accumCol += mask * Get_Sky_Color(r, normalize(sunDirection)); // add reflective result to the refractive result (if any)
				break;
			}
			
			// reached the sky light, so we can exit
			break;
		} // end if (t == INFINITY)
		
		
		// useful data 
		n = normalize(intersec.normal);
                nl = dot(n, r.direction) < 0.0 ? normalize(n) : normalize(-n);
		x = r.origin + r.direction * t;
		
		if (bounces == 0) 
			firstX = x;

		// ray hits terrain
		if (intersec.type == TERRAIN)
		{
			float rockNoise = texture(t_PerlinNoise, (0.0001 * x.xz) + 0.5).x;
			vec3 rockColor0 = vec3(0.2, 0.2, 0.2) * 0.01 * rockNoise;
			vec3 rockColor1 = vec3(0.2, 0.2, 0.2) * rockNoise;
			vec3 snowColor = vec3(0.9);
			vec3 up = normalize(vec3(0, 1, 0));
			vec3 randomSkyVec = randomCosWeightedDirectionInHemisphere(mix(n, up, 0.9));
			vec3 skyColor = Get_Sky_Color( Ray(x, randomSkyVec), uSunDirection );
			if (dot(randomSkyVec, uSunDirection) > 0.98) skyColor *= 0.01;
			vec3 sunColor = clamp(Get_Sky_Color( Ray(x, randomDirectionInSpecularLobe(uSunDirection, 0.1)), uSunDirection ), 0.0, 2.0);
			float terrainLayer = clamp( ((x.y + -TERRAIN_LIFT) + (rockNoise * 1000.0) * n.y) / (TERRAIN_HEIGHT * 1.2), 0.0, 1.0 );
			
			if (x.y > uWaterLevel && terrainLayer > 0.95 && terrainLayer > 0.9 - n.y)
			{
				intersec.color = mix(vec3(0.5), snowColor, n.y);
				mask = mix(intersec.color * skyColor, intersec.color * sunColor, 0.7);// ambient color from sky light
			}	
			else
			{
				intersec.color = mix(rockColor0, rockColor1, clamp(terrainLayer * n.y, 0.0, 1.0) );
				mask = intersec.color * skyColor;
				if (x.y > uWaterLevel && cameraUnderWaterValue == 0.0 && bounces == 0)
				{
					nc = 1.0; // IOR of air
					nt = 1.2; // IOR of watery rock
					Re = calcFresnelReflectance(r.direction, n, nc, nt, ratioIoR);
					Tr = 1.0 - Re;
					firstTypeWasREFR = true;
					reflectionTime = false;
					firstRay = Ray( x, reflect(r.direction, n) );
					firstRay.origin += nl * uEPS_intersect;
					mask *= Tr;
					firstMask = vec3(1) * Re;
				}
			}
				
			
			vec3 shadowRayDirection = randomDirectionInSpecularLobe(uSunDirection, 0.4);						
			if ( isLightSourceVisible(x, n, shadowRayDirection) && x.y > uWaterLevel ) // in direct sunlight
			{
				mask = intersec.color * sunColor;	
			}

			if (rayEnteredWater)
			{
				rayEnteredWater = false;
				mask *= exp(log(WATER_COLOR) * thickness * t); 
			}
				
			if (firstTypeWasREFR )
			{
				if (!reflectionTime) 
				{	
					accumCol = mask;
					// start back at the refractive surface, but this time follow reflective branch
					r = firstRay;
					r.direction = normalize(r.direction);
					mask = firstMask;
					// set/reset variables
					reflectionTime = true;
					// continue with the reflection ray
					continue;
				}

				accumCol += mask;
				break;
			}

			accumCol = mask;	
			break;
		} // end if (intersec.type == TERRAIN)

		
		if (intersec.type == REFR)  // Ideal dielectric REFRACTION
		{
			nc = 1.0; // IOR of air
			nt = 1.33; // IOR of water
			Re = calcFresnelReflectance(r.direction, n, nc, nt, ratioIoR);
			Tr = 1.0 - Re;
			
			if (bounces == 0)
			{	
				// save intersection data for future reflection trace
				firstTypeWasREFR = true;
				firstMask = mask * Re;
				firstRay = Ray( x, reflect(r.direction, nl) ); // create reflection ray from surface
				firstRay.origin += nl * uEPS_intersect;
				mask *= Tr;
				rayEnteredWater = true;
			}
			
			// transmit ray through surface
			
			// is ray leaving a solid object from the inside? 
			// If so, attenuate ray color with object color by how far ray has travelled through the medium
			if (distance(n, nl) > 0.1)
			{
				rayEnteredWater = false;
				mask *= exp(log(WATER_COLOR) * thickness * t);
			}
			
			tdir = refract(r.direction, nl, ratioIoR);
			r = Ray(x, normalize(tdir));
			r.origin -= nl * uEPS_intersect;
			
			continue;
			
		} // end if (intersec.type == REFR)
		
	} // end for (int bounces = 0; bounces < 3; bounces++)
	
	
	// atmospheric haze effect (aerial perspective)
	float fogStartDistance = TERRAIN_FAR * 0.3;
	float hitDistance = distance(cameraRay.origin, firstX);
	float fogDistance;

	if (skyHit && cameraUnderWaterValue == 0.0 ) // sky and clouds
	{
		//vec3 cloudColor = cld.rgb / (cld.a + 0.00001);
		//vec3 sunColor = clamp(Get_Sky_Color( Ray(skyPos, normalize((randVec * 0.03) + sunDirection)), sunDirection ), 0.0, 1.0);
		//cloudColor *= mix(sunColor, vec3(1), max(0.0, dot(vec3(0,1,0), sunDirection)) );
		//cloudColor = mix(initialSkyColor, cloudColor, clamp(cld.a, 0.0, 1.0));
		//accumCol = mask * mix( accumCol, cloudColor, clamp( exp2( -hitDistance * 0.003 ), 0.0, 1.0 ) );
		fogDistance = max(0.0, hitDistance - fogStartDistance);
		accumCol = mix( initialSkyColor, accumCol, clamp( exp(-(fogDistance * 0.00005)), 0.0, 1.0 ) );
	}	
	else // terrain and other objects
	{
		fogDistance = max(0.0, hitDistance - fogStartDistance);
		accumCol = mix( initialSkyColor, accumCol, clamp( exp(-(fogDistance * 0.00005)), 0.0, 1.0 ) );

		// underwater fog effect
		hitDistance *= cameraUnderWaterValue;
		accumCol = mix( vec3(0.0,0.001,0.001), accumCol, clamp( exp2( -hitDistance * 0.001 ), 0.0, 1.0 ) );
	}
	
	
	return max(vec3(0), accumCol); // prevents black spot artifacts appearing in the water 
	     
}

/*
//-----------------------------------------------------------------------
void SetupScene(void)
//-----------------------------------------------------------------------
{
	vec3 z  = vec3(0);// No color value, Black        
	
}
*/

// tentFilter from Peter Shirley's 'Realistic Ray Tracing (2nd Edition)' book, pg. 60		
float tentFilter(float x)
{
	return (x < 0.5) ? sqrt(2.0 * x) - 1.0 : 1.0 - sqrt(2.0 - (2.0 * x));
}

void main( void )
{
	// not needed, three.js has a built-in uniform named cameraPosition
	//vec3 camPos   = vec3( uCameraMatrix[3][0],  uCameraMatrix[3][1],  uCameraMatrix[3][2]);
	
    	vec3 camRight   = vec3( uCameraMatrix[0][0],  uCameraMatrix[0][1],  uCameraMatrix[0][2]);
    	vec3 camUp      = vec3( uCameraMatrix[1][0],  uCameraMatrix[1][1],  uCameraMatrix[1][2]);
	vec3 camForward = vec3(-uCameraMatrix[2][0], -uCameraMatrix[2][1], -uCameraMatrix[2][2]);
	
	// calculate unique seed for rng() function
	seed = uvec2(uFrameCounter, uFrameCounter + 1.0) * uvec2(gl_FragCoord); // old way of generating random numbers

	randVec4 = texture(tBlueNoiseTexture, (gl_FragCoord.xy + (uRandomVec2 * 255.0)) / 255.0); // new way of rand()
	
	vec2 pixelOffset = vec2( tentFilter(rng()), tentFilter(rng()) ) * 0.5;
	// we must map pixelPos into the range -1.0 to +1.0
	vec2 pixelPos = ((gl_FragCoord.xy + pixelOffset) / uResolution) * 2.0 - 1.0;

	vec3 rayDir = normalize( pixelPos.x * camRight * uULen + pixelPos.y * camUp * uVLen + camForward );
	
	// depth of field
	vec3 focalPoint = uFocusDistance * rayDir;
	float randomAngle = rand() * TWO_PI; // pick random point on aperture
	float randomRadius = rand() * uApertureSize;
	vec3  randomAperturePos = ( cos(randomAngle) * camRight + sin(randomAngle) * camUp ) * sqrt(randomRadius);
	// point on aperture to focal point
	vec3 finalRayDir = normalize(focalPoint - randomAperturePos);
    
	Ray ray = Ray( cameraPosition + randomAperturePos, finalRayDir );

	//SetupScene(); 

	// perform path tracing and get resulting pixel color
	vec3 pixelColor = CalculateRadiance(ray, uSunDirection);
	
	vec3 previousColor = texelFetch(tPreviousTexture, ivec2(gl_FragCoord.xy), 0).rgb;
	
	if ( uCameraIsMoving )
	{
		previousColor *= 0.8; // motion-blur trail amount (old image)
		pixelColor *= 0.2; // brightness of new image (noisy)
	}
	else
	{
		previousColor *= 0.9; // motion-blur trail amount (old image)
		pixelColor *= 0.1; // brightness of new image (noisy)
	}
	
	
	pc_fragColor = vec4( pixelColor + previousColor, 1.0 );	
}
