/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <climits>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;
default_random_engine gen;
bool debug = false;

void ParticleFilter::init(double x, double y, double theta, double std[]) 
{
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  
  //default_random_engine gen;
  num_particles  = 300;
  
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];
  
  // Create gaussian distribution for noise, centered around the GPS x and y co-ordinates and theta
  normal_distribution<double> noise_x(0, std_x);
  normal_distribution<double> noise_y(0, std_y);
  normal_distribution<double> noise_theta(0, std_theta);
  
  for (int i = 0; i < num_particles; i++)
  {
    Particle p_i;
    p_i.id = i;
    p_i.x = x;
    p_i.y = y;
    p_i.theta = noise_theta(gen);
    p_i.weight = 1.0;
    
    p_i.x += noise_x(gen);
    p_i.y += noise_y(gen);
    p_i.theta += noise_theta(gen);
    
    particles.push_back(p_i);
    weights.push_back(p_i.weight);
    
    if(debug)
    {
      cout << "Initial Particle no. "<< i <<": " << p_i.x <<"," <<p_i.y <<","<< p_i.theta << endl;
      cout << "Initialized!!" << endl;
    }
  }
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) 
{
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  
  //default_random_engine gen;
  
  normal_distribution<double> noise_x(0, std_pos[0]);
  normal_distribution<double> noise_y(0, std_pos[1]);
  normal_distribution<double> noise_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++)
  {
    // Predict new state
    if (fabs(yaw_rate) < 0.00001)
    {
      particles[i].x += velocity * delta_t * cos(particles[i].theta) + noise_x(gen);
      particles[i].y += velocity * delta_t * sin(particles[i].theta) + noise_y(gen);
      particles[i].theta += noise_theta(gen);
    }
    else
    {
      particles[i].x += (velocity/yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)) + noise_x(gen);
      particles[i].y += (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t)) + noise_y(gen);
      particles[i].theta += yaw_rate * delta_t + noise_theta(gen);
    }
    
    if(debug)
    {
      cout << "---------------------Prediction Calc------------------------" << endl;
      cout << "Velocity:" << velocity << "; Yawrate:" << yaw_rate << endl;
      cout << "Predicted:" << particles[i].x << " , " << particles[i].y << " , " << particles[i].theta << endl;
    }
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) 
{
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  double p2lm_dist;
  
  for (int i=0; i < observations.size(); i++)
  { // loop over observations
    double min_dist = INT_MAX;
    int closest_map_id = -1;
    
    for (int j=0; j < predicted.size(); j++)
    {
      p2lm_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      
      if (p2lm_dist < min_dist)
      {
        min_dist = p2lm_dist;
        closest_map_id = predicted[j].id;
      }
    }
    observations[i].id  = closest_map_id;
    
    if(debug)
    {
      //Found the smallest distance. Now asssign the values of the nearest prediction to the observation
      cout << "Landmark Index:" << observations[i].id << endl;
      cout << "TObservation(x,y): " << "("<<observations[i].x << "," << observations[i].y << ") ;" << endl;
      cout << "Predicted(x,y): " << "(" << predicted[closest_map_id].x << "," << predicted[closest_map_id].y << ")"
              << "Dist:" << min_dist << "\n" << endl;
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   std::vector<LandmarkObs> observations, Map map_landmarks)
{
	// TODO: Update the weights of each particle using a multi-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 3.33
	//   http://planning.cs.uiuc.edu/node99.html
  
  for (int p_num = 0; p_num < num_particles; p_num++)
  {
    // Extract particle x and y co-ordinates
    double part_x = particles[p_num].x;
    double part_y = particles[p_num].y;
    double part_theta = particles[p_num].theta;
    
    // These are the observations within sensor range from the predicted particle position to map landmarks
    vector<LandmarkObs> predicted_obs_in_range;
    
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++)
    {
      // Extract map landmark ids and x,y coordinates
      int lm_id = map_landmarks.landmark_list[j].id_i;
      float lm_x = map_landmarks.landmark_list[j].x_f;
      float lm_y = map_landmarks.landmark_list[j].y_f;
      
      // Use only landmarks which are within sensor range
      if (dist(part_x, part_y, lm_x, lm_y) <= sensor_range)
      {
        predicted_obs_in_range.push_back(LandmarkObs{lm_id, lm_x, lm_y});
      }
    }
    
    // Transform observations from Vehicle co-ordinates to Map co-ordinates
    vector<LandmarkObs> transform_car2map;
    
    for (unsigned int k = 0; k < observations.size(); k++)
    {
      double t_x = observations[k].x * cos(part_theta) - observations[k].y * sin(part_theta) + part_x;
      double t_y = observations[k].x * sin(part_theta) + observations[k].y * cos(part_theta) + part_y;
      transform_car2map.push_back(LandmarkObs{observations[k].id, t_x, t_y});
    }
  
    dataAssociation(predicted_obs_in_range, transform_car2map);
    
    // Reinitialize particle weight
    particles[p_num].weight = 1.0;
    
    for (unsigned int m = 0; m < transform_car2map.size(); m++)
    {
      // Placeholders for observation and associated prediction coordinates
      double ob_x, ob_y, pr_x, pr_y;
      ob_x = transform_car2map[m].x;
      ob_y = transform_car2map[m].y;
      
      int associated_prediction = transform_car2map[m].id;
      
      // Get the x,y coordinates of the prediction associated with the current observation
      for (unsigned int n = 0; n < predicted_obs_in_range.size(); n++)
      {
        if (predicted_obs_in_range[n].id == associated_prediction)
        {
          pr_x = predicted_obs_in_range[n].x;
          pr_y = predicted_obs_in_range[n].y;
        }
      }
    
      // Calculate weight for this observation with multivariate Gaussian
      double sd_x = std_landmark[0];
      double sd_y = std_landmark[1];
      double obs_w = ( 1 /(2 * M_PI * sd_x * sd_y)) * exp( -( pow(ob_x - pr_x, 2)/(2 * pow(sd_x, 2)) + (pow(ob_y - pr_y, 2)/(2 * pow(sd_y, 2)))));
      
      // product of this obersvation weight with total observations weight
      particles[p_num].weight *= obs_w;
    }
  }
}

void ParticleFilter::resample()
{
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  
  /*discrete_distribution<int> dist_weights {weights.begin(), weights.end()};
  vector<Particle> new_particles;
  
  for (int i=0; i < num_particles; i++)   // i is just a counter
  {
    int new_particle_index = dist_weights(gen);
    Particle new_particle = particles[new_particle_index];
    new_particles.push_back(new_particle);
  }
  particles = new_particles;
  */
  
  vector<Particle> new_particles;
  
  // get all of the current weights
  vector<double> weights;
  for (int i = 0; i < num_particles; i++)
  {
    weights.push_back(particles[i].weight);
  }
  
  // generate random starting index for resampling wheel
  uniform_int_distribution<int> uniintdist(0, num_particles-1);
  auto index = uniintdist(gen);
  
  // get max weight
  double max_weight = *max_element(weights.begin(), weights.end());
  
  // uniform random distribution [0.0, max_weight)
  uniform_real_distribution<double> unirealdist(0.0, max_weight);
  
  double beta = 0.0;
  
  // spin the resample wheel!
  for (int i = 0; i < num_particles; i++)
  {
    beta += unirealdist(gen) * 2.0;
    while (beta > weights[index])
    {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }
  
  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
