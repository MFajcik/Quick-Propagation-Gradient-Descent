/**
* Demonstration of Quick propagation algorithm
* Author: Bc.Martin Fajčík xfajci00@stud.fit.vutbr.cz
* Institution: VUT FIT
* Course: Soft Computing (SFC) 2015
*/

#ifndef _HEADER_INCLUDE
#define _HEADER_INCLUDE

#include <SDL.h>
#include <cairo.h>
#include <vector>
#include <limits>
#include <iostream>
#include <cmath>
#include <ctime>
#include <random>
#include <fstream>
#include <sstream>
#include <thread>
#include <iomanip>
#include <atomic>


// defined colors
#define WHITE 0xffffffff

//Warnings
#define UNKNOWN_CONFIG_PARAMETER 1
#define FILE_OPEN_FAILED 2
#define INVALID_NEURONCOUNT_VALUE 3
#define IMPROPER_CONFIGURATION 4
#define CONFIG_NOT_LOADED 5
#define NN_NOTFOUND 6
#define WRONG_INPUT_SIZE 7
#define WRONG_CONFIG_FORMAT 8

//windows cairo devpack does not contain M_PI preprocessor constant
#ifndef M_PI
#define M_PI 3.14159265
#endif

//render info
#define ITERATIONX 500
#define ITERATIONY 580
#define EXITINFOX 5
#define EXITINFOY 20
#define NN_WIDTH 600
#define MAX_NEURON_DIAMETER 50
#define NN_HEIGHT 400
#define MAX_FONT_SIZE 22.0
#define RED_RGBA 1.0, 0.2, 0.3, 1.0
#define BLACK_RGBA 0.0,0.0,0.0,1.0

//global variables
bool configloaded = false;
bool dataloaded = false;
SDL_Surface *screen;
SDL_Event event;
std::atomic<bool> window_closed;

bool  demo_bpqp = false, visualize = false,breaklearning;
const double EulerConstant = exp(1.0);
const double PI2 = 2 * M_PI;

namespace qp //set already has different semantics in std namespace
{
	class set
	{
	private:
		double * _x;
		double * _d;
	public:
		set(double* x, double* d)
		{
			_x = x;
			_d = d;
		}
		double* getX()
		{
			return _x;
		}
		double* getD()
		{
			return _d;
		}
	};
}


class TrainingSet
{
    std::vector<qp::set*> trainingsetlist;
public:
    TrainingSet ()
    {
    }
    void add (double* x, double* d)
    {
        trainingsetlist.push_back(new qp::set(x,d));
    }
    void clear()
    {
        for(qp::set* i : trainingsetlist)
        {
            delete[] i->getX();
            delete[] i->getD();
        }
        trainingsetlist.clear();
    }
    qp::set* get (unsigned i)
    {
        return trainingsetlist[i];
    }
    unsigned getLength ()
    {
        return trainingsetlist.size();
    }
};

typedef struct position_s
{
    int x,y;
} position;

typedef struct line_s
{
    position p1,p2;
    unsigned x,y,z;
	double angle;
} connection;

typedef struct circle_s
{
    position p, textpos, nnpos;
    std::string text;
    unsigned radius;
} neuron;

typedef struct scene_s
{
    std::vector<connection*> connections;
    std::vector<neuron*> neurons;

    void destroy()
    {
        connections.clear();
        neurons.clear();
    }
} scene;


typedef struct config_s
{
    double  threshold,mi, momentum, lambda, epsilon;
    unsigned layercount, inputs, maxiterations;
    unsigned* neurons;
    void copy(config_s* c)
    {
        c->threshold = threshold;
        c->momentum = momentum;
        c->lambda = lambda;
        c->mi = mi;
        c->epsilon = epsilon;
        c->layercount = layercount;
        c->inputs = inputs;
        c->maxiterations = maxiterations;

        c->neurons = new unsigned[layercount+1];
        for (unsigned i =0; i<=layercount; i++)
            c->neurons[i]=neurons[i];
    }
}
Config;

class QPNetwork
{
public:
    double _mi;//learning rate
    double _momentum;//accelerating constant - used to pass local minimums
    double _shrink;//shrink factor (elemination of very small quickprop steps)
    double _lambda;//sigmoid "sharpness"
    double _epsilon;//maximal step

    unsigned L;//number of layers(without zero "input" layer)
    unsigned* n;//number of neurons in each layer
    double threshold;
    double maxiterations;

    //enumerators
    unsigned long iter;
    unsigned long QPJ_count;

	bool networkallocated;

    double GLerr;//global network error
    double** y;//neuron outputs (inputs are placed in zero "input" layer)
    double** delta;//differentiated error for hidden layers
    double*** w;//input weights
    //+1 because of zero layer
    double*** old_gradient;//last iteration error gradient
    double*** gradient;//error gradient (following the direction of gradient descent)
    double*** old_step;//old step towards E minimum
    //step towards E minimum
    //in fact it is an error gradient multiplied by learning rate (_mi)
    double*** step;

    QPNetwork(Config* config);
    ~QPNetwork() ;
    void learn(double** data, unsigned samples, bool run_bp);
    void printlearningstatistics();
    void validate_learning();
    void qp_step(unsigned l, unsigned j, unsigned i);
    void bp_step(unsigned l, unsigned j, unsigned i);
    void feed_forward();
    void init(double ** data, TrainingSet* ts, unsigned samples);
    void allocate_network();
    void summarize_information();
    void back_propagate();
    void restart(Config* config);
    void setconfig(Config* config);
    void deallocate_network();
    void test(double** data, unsigned samples);
};

QPNetwork::QPNetwork(Config* config)
{
	
    networkallocated = false;
     _mi = config->mi;
    _lambda = config->lambda;
    _momentum = config->momentum;
    _epsilon = config->epsilon;
    _shrink = _epsilon/(1.0+_epsilon);

    L = config->layercount;
    n = config->neurons;
    maxiterations = config->maxiterations;
    threshold = config->threshold;

    allocate_network();
}

QPNetwork::~QPNetwork()
{
    if (networkallocated) deallocate_network();
    if (configloaded) delete n;
}

double uniform_random(double lower_bound, double upper_bound );
void printInfo();
void printHelp();
void error (unsigned number);
void warning(unsigned number);
bool read_configuration(Config* config);
bool read_datafile(double*** data, unsigned * matrix_height, unsigned * matrix_width);
void run_program_loop();
char readoption();
void precalculate(scene* s,QPNetwork* qp_nn);
#endif
