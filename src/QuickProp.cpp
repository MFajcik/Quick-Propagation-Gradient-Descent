/**
* Demonstration of Quick propagation algorithm
* Author: Bc.Martin Fajčík
* Institution: VUT FIT
* Course: Soft Computing (SFC) 2015
*/

#include "neuron.h"

/** Uniform distribution of random numbers */
double uniform_random(double lower_bound, double upper_bound )
{
    static std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
    static std::default_random_engine generator(unsigned(time(nullptr)));
    return unif(generator);
}

/**Scene Rendering*/
void draw (QPNetwork* qp_nn, scene* s)
{
    cairo_t *cr;
    cairo_surface_t *cairosurf;

    cairosurf = cairo_image_surface_create_for_data(
                    (unsigned char*)screen->pixels,
                    CAIRO_FORMAT_ARGB32,
                    screen->w,
                    screen->h,
                    (Uint32)screen->pitch);

    cr = cairo_create(cairosurf);
    SDL_FillRect(screen, NULL, WHITE);

    cairo_set_line_cap(cr,CAIRO_LINE_CAP_ROUND);
    cairo_select_font_face(cr, "Georgia", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_line_width (cr, 2.0);

	//draw neurons and their outputs
    for (neuron*c:s->neurons)
    {
        cairo_set_source_rgba (cr, BLACK_RGBA);
        cairo_arc(cr, c->p.x, c->p.y, c->radius, 0, 2 * M_PI);
        cairo_stroke (cr);

        std::ostringstream strs;
        strs << std::fixed<< std::setprecision(2)<<qp_nn->y[c->nnpos.x][c->nnpos.y];
        c->text = strs.str();

        cairo_text_extents_t te;
        unsigned text_x = c->p.x;
        unsigned text_y = c->p.y;
        double fontSize = MAX_FONT_SIZE;
        while (1)
        {
            cairo_set_font_size(cr, fontSize);
            cairo_text_extents(cr, c->text.c_str(), &te);
            if (te.width < (c->radius*1.6))  // Allow a little room on each side 1.6=2*0.8 ,2 is for diameter and then 80% of space of circle, which is 0.8
            {
                // Calculate the position
                text_x -=(unsigned)(te.width/2.0);
                text_y +=(unsigned)(te.height/2.0);
                break;
            }
            fontSize --;
            if (fontSize <1.0)
            {
                cairo_set_font_size(cr, 1.0);
                break;
            }
        }
        cairo_set_source_rgba (cr, RED_RGBA);
        cairo_move_to(cr,text_x,text_y);
        cairo_show_text(cr, c->text.c_str());
        cairo_stroke (cr);
    }

	//draw connections and wieghts above
    for (connection* c: s->connections)
    {
        cairo_set_source_rgba (cr, BLACK_RGBA);
        cairo_move_to(cr, c->p1.x, c->p1.y);
        cairo_line_to (cr, c->p2.x, c->p2.y);
        cairo_stroke (cr);
		//this calculation must be inside UI thread
        if (c->x!=std::numeric_limits<unsigned>().max())
        {
            cairo_set_source_rgba (cr, RED_RGBA);
            std::ostringstream strs;
            strs << std::fixed<< std::setprecision(2)<<qp_nn->w[c->x][c->y][c->z];
            std::string weight = strs.str();

            cairo_text_extents_t te;
            cairo_font_extents_t fe;
            cairo_set_font_size(cr, MAX_FONT_SIZE/2);
            cairo_save(cr);
            cairo_font_extents(cr, &fe);
            cairo_text_extents(cr, weight.c_str(), &te);
            cairo_translate(cr, c->p1.x+(c->p2.x-c->p1.x)*0.75, c->p1.y+(c->p2.y-c->p1.y)*0.75);
            cairo_rotate(cr, c->angle);
            cairo_translate(cr, -te.width/2, -fe.height/2);
            cairo_move_to(cr,0,0);
            cairo_show_text(cr, weight.c_str());
            cairo_restore(cr);
            cairo_stroke(cr);
        }
    }

	//write info
	//iterations
    cairo_set_font_size(cr, MAX_FONT_SIZE);
    cairo_set_source_rgba (cr, BLACK_RGBA);	
    std::ostringstream strs;
	strs <<qp_nn->iter;    
	std::string iteration = "Iteration: "+strs.str();
	cairo_move_to(cr,ITERATIONX,ITERATIONY);
    cairo_show_text(cr, iteration.c_str());

	//exit info
	cairo_move_to(cr,EXITINFOX,EXITINFOY);
    cairo_set_font_size(cr, MAX_FONT_SIZE/2);
    cairo_show_text(cr, "Press S to close the window");

    cairo_destroy(cr);
    SDL_Flip(screen);
}


// SDL timer generating userevents - 1 userevent = 1 redraw of scene
static Uint32 timer(Uint32 inter, void *param)
{
    SDL_Event event;
    (void)param;
    event.type = SDL_USEREVENT;
    SDL_PushEvent (&event);
    return inter;
}

void display_algorithm(QPNetwork* qp_nn)
{
    window_closed = false;
    // initialize SDL
    if (SDL_Init(SDL_INIT_EVERYTHING) < 0)
    {
        std::cerr << "ERROR: Unable to initialize SDL: " << SDL_GetError() << std::endl;
        return;
    }
    atexit(SDL_Quit);
    // open a screen with the specified properties
    screen = SDL_SetVideoMode(800, 600, 32, SDL_SWSURFACE);
    if (screen == NULL)
    {
        std::cerr << "Unable to set " << "800" << 'x' << "600" << "video: "
                  << SDL_GetError() << std::endl;
        return;
    }
    // set windows properties
    SDL_WM_SetCaption("SFC - Demonstration of Quick propagation algorithm", "ico");
    SDL_AddTimer(1, timer, NULL);
    scene s;
    precalculate(&s, qp_nn);
    draw( qp_nn,  &s);

    bool drawingdone = false,drawlastframe = false;
    while(SDL_WaitEvent(&event) && !drawingdone)
    {
        switch(event.type)
        {
        case SDL_USEREVENT:
            if (!breaklearning)
                draw( qp_nn,  &s);
            else if (!drawlastframe)//make sure network drawn is always in final state
            {
                draw( qp_nn,  &s);
                drawlastframe = true;
            }
            break;
        case SDL_VIDEORESIZE:
            screen = SDL_SetVideoMode(event.resize.w,
                                      event.resize.h, 32,
                                      SDL_HWSURFACE|SDL_RESIZABLE);
            break;
        case SDL_KEYDOWN:
            if(event.key.keysym.sym == SDLK_s)
            {
                drawingdone = true;
            }
            break;
        case SDL_QUIT:
            drawingdone = true;
        }
    }
    s.destroy();
    SDL_Quit();
    window_closed = true;
}

void QPNetwork::allocate_network()
{
    y = new double*[L + 1]();
    delta = new double*[L + 1]();

    for (unsigned i = 0; i < L + 1; i++)
    {
        y[i] = new double[n[i] + 1]();//+1 is BIAS
        delta[i] = new double[n[i] + 1]();
    }
    old_gradient = new double**[L + 1]();
    old_step = new double**[L + 1]();
    gradient = new double**[L + 1]();
    w = new double**[L + 1]();
    step = new double**[L + 1]();
    for (unsigned l = 0; l < L + 1; l++)
    {
        w[l] = new double*[n[l] + 1]();
        old_gradient[l] = new double*[n[l] + 1]();
        gradient[l] = new double*[n[l] + 1]();
        old_step[l] = new double*[n[l] + 1]();
        step[l] = new double*[n[l] + 1]();
        for (unsigned j = 0; j < n[l] + 1; j++)
        {
            if (l > 0)
            {
                w[l][j] = new double[n[l - 1] + 1]();
                old_gradient[l][j] = new double[n[l - 1] + 1]();
                gradient[l][j] = new double[n[l - 1] + 1]();
                old_step[l][j] = new double[n[l - 1] + 1]();
                step[l][j] = new double[n[l - 1] + 1]();
                for (unsigned i = 0; i < n[l - 1] + 1; i++)
                {
                    w[l][j][i] = 0.1- uniform_random(std::numeric_limits<double>().min(), 200.0) / 1000.0;
                    step[l][j][i] = 0.0;
                }
            }
            else
            {
                w[l][j] = new double[n[0] + 1]();
                old_gradient[l][j] = new double[n[0] + 1]();
                gradient[l][j] = new double[n[0] + 1]();
                old_step[l][j] = new double[n[0] + 1]();
                step[l][j] = new double[n[0] + 1]();
                for (unsigned i = 0; i < n[0] + 1; i++)
                {
                    w[l][j][i] =  0.1 - uniform_random(std::numeric_limits<double>().min(), 200.0) / 1000.0;
                    step[l][j][i] = 0.0;
                }
            }
        }
    }
    this->networkallocated = true;
}
void QPNetwork::deallocate_network()
{
    if (!networkallocated)
        return;
    for (unsigned i = 0; i < L + 1; i++)
    {
        delete[] y[i];
        delete[] delta[i];
    }
    delete[] y;
    delete[] delta;
    for (unsigned l = 0; l < L + 1; l++)
    {
        for (unsigned j = 0; j < n[l] + 1; j++)
        {
            delete[] w[l][j];
            delete[] gradient[l][j];
            delete[] old_gradient[l][j];
            delete[] step[l][j];
            delete[] old_step[l][j];
        }
        delete[] w[l];
        delete[] gradient[l];
        delete[] old_gradient[l];
        delete[] step[l];
        delete[] old_step[l];
    }
    delete[] w;
    delete[] gradient;
    delete[] old_gradient;
    delete[] step;
    delete[] old_step;

    this->networkallocated = false;
}
void QPNetwork::init(double ** data,TrainingSet* ts, unsigned samples)
{
    //initialize training set
    for (unsigned i = 0; i < samples; i++)
    {
        double* i_n = new double[n[0]];
        double* o_n = new double[n[L]];
        for (unsigned k = 0; k < n[0]; k++)
            i_n[k] = data[i][k];
        for (unsigned k = n[0]; k < n[0]+n[L]; k++)
            o_n[k- n[0]] = data[i][k];
        ts->add(i_n, o_n);
    }
}
void QPNetwork::feed_forward()
{
    for (unsigned l = 1; l <= L; l++)
        for (unsigned j = 1; j <= this->n[l]; j++)
        {
            double u = 0.0;
            for (unsigned i = 0; i <= n[l - 1]; i++)
                u += w[l][j][i] * y[l - 1][i];
            y[l][j] = 1.0 / (1.0 + pow(EulerConstant, -(_lambda * u)));
        }
}
/** Single thread checks if S is pressed during calculation via this function*/
void CheckKeyPressed(bool* breaklearning)
{
    char option =0;
    while (option!='S' && option != 's' && !(*breaklearning))
        option = getchar();
    *breaklearning = true;
}

void QPNetwork::learn(double** data, unsigned samples, bool run_bp)
{
    TrainingSet ts;
    init(data, &ts,samples);
    iter = 0;
    QPJ_count = 0;
    breaklearning = false;
    std::cout<<"Press S to stop machine learning.\n";
    std::thread keypressed(CheckKeyPressed, &breaklearning);
    std::cout<<std::setprecision(15);
    do
    {
        iter++;
        std::cout<<"Iteration: "<<iter<<" Global error: "<<GLerr<<"\r";
        this->GLerr = 0.0;
        unsigned bp_steps = 0;
        for (unsigned p = 0; p < ts.getLength(); p++)
        {

            qp::set* t_element = ts.get(p);

            //prepare 1 into "zero" inputs for neurons in each layer
            for (unsigned l = 0; l < L; l++)
                y[l][0] = 1.0;

            //prepare inputs into "zero" layer output
            double* input = t_element->getX();
            for (unsigned i = 1; i <= n[0]; i++)
                y[0][i] = input[i - 1];

            //feed forward network - produce output for neuron network
            this->feed_forward();

            //calculate network (last layer) mean squared error
            double E = 0.0;
            for (unsigned j = 1; j <= n[L]; j++)
            {
                double err = t_element->getD()[j-1] - y[L][j];//-1 because data in training set are indexed from 0, they does not contain initial 1
                E += err * err;
                delta[L][j] = err * _lambda * y[L][j] * (1.0 - y[L][j]);
            }

            GLerr += 0.5 * E;
            //backpropagating error into hidden layers
            this->back_propagate();

            //adjusting weights
            for (unsigned l = 1; l <= L; l++){
                for (unsigned j = 1; j <= n[l]; j++)
                    for (unsigned i = 0; i <= n[l - 1]; i++)
                        if (bp_steps <2)  // Do first two iterations with back-propagation
                        {
                            this->bp_step(l,j,i);                            
							if (!run_bp)
								bp_steps++;
                        }
                        else //do large steps - Quick propagation
                            this->qp_step(l,j,i);
			}
        }
    }
    while (GLerr > threshold && iter < maxiterations && !breaklearning);
    if (!breaklearning)
        std::cout<<"\nPress Enter to continue.\n";
    breaklearning = true;
    keypressed.join();
    std::cout<<"\n";
    ts.clear();
}
void QPNetwork::back_propagate()
{
    for (int l = L - 1; l > 0; l--)
        for (unsigned j = 1; j <= n[l]; j++)
        {
            double summ = 0.0;
            for (unsigned k = 1; k <= n[l + 1]; k++)
                summ += delta[l + 1][k] * w[l + 1][k][j];
            delta[l][j] = summ * _lambda * y[l][j] * (1.0 - y[l][j]);
        }
}
void QPNetwork::bp_step(unsigned l, unsigned j, unsigned i)
{
    old_gradient[l][j][i] = gradient[l][j][i];
    gradient[l][j][i] = delta[l][j] * y[l - 1][i];

    old_step[l][j][i] = step[l][j][i] ;
    step[l][j][i] = _mi *gradient[l][j][i];

    w[l][j][i] += step[l][j][i]+ _momentum*old_step[l][j][i];
}
void QPNetwork::qp_step(unsigned l, unsigned j, unsigned i)
{
    old_gradient[l][j][i] = gradient[l][j][i];
    gradient[l][j][i] = delta[l][j] * y[l - 1][i];
    old_step[l][j][i] = step[l][j][i];
    step[l][j][i] = _epsilon *gradient[l][j][i];

    double dstep = 0.0;

    if (step[l][j][i]<0.0)
    {
        if (gradient[l][j][i]<0.0)
            dstep+=_epsilon*gradient[l][j][i];
        if (gradient[l][j][i]<=_shrink*old_gradient[l][j][i])
            dstep += _mi*step[l][j][i];
        else
        {
            dstep+=old_step[l][j][i] * (gradient[l][j][i] / (gradient[l][j][i] - old_gradient[l][j][i]));
            QPJ_count++;
        }
    }
    else if (step[l][j][i]>=0.0)
    {
        if (gradient[l][j][i]>0.0)
            dstep+=_epsilon*gradient[l][j][i];
        if (gradient[l][j][i]>=_shrink*old_gradient[l][j][i])
            dstep += _mi*step[l][j][i];
        else
        {
            dstep+=old_step[l][j][i] * (gradient[l][j][i] / (gradient[l][j][i] - old_gradient[l][j][i]));
            QPJ_count++;
        }
    }
    else
        dstep += _epsilon*gradient[l][j][i];
    w[l][j][i] += dstep;
}
void QPNetwork::test(double** data, unsigned samples)
{
    TrainingSet ts;
    //initialize training set
    for (unsigned i = 0; i < samples; i++)
    {
        double* i_n = new double[n[0]];
        for (unsigned k = 0; k < n[0]; k++)
            i_n[k] = data[i][k];

        //0.0 - xpected output is irrelevant, because we are testing network now.
        ts.add(i_n, nullptr);
    }

    this->GLerr = 0.0;

    for (unsigned p = 0; p < ts.getLength(); p++)
    {
        for (unsigned l = 0; l < L; l++)
            y[l][0] = 1.0;

        qp::set* t_element = ts.get(p);
        double* input = t_element->getX();
        for (unsigned i = 1; i < n[0] + 1; i++)
            y[0][i] = input[i - 1];

        std::cout<<"INPUTS:\n";
        for (unsigned x = 1; x<=n[0]; x++)
            std::cout<<y[0][x]<< " ";
        std::cout<<"\nOUTPUTS:\n";

        for (unsigned l = 1; l <= L; l++)
            for (unsigned j = 1; j <= this->n[l]; j++)
            {
                double u = 0.0;
                for (unsigned i = 0; i <= n[l - 1]; i++)
                    u += w[l][j][i] * y[l - 1][i];
                y[l][j] = 1.0 / (1.0 + pow(EulerConstant, -(_lambda * u)));
                if (l == L)
                    std::cout<<" = "<<y[l][j]<<std::endl;
            }
    }
    ts.clear();
}

void printInfo()
{
    std::cout<<"Demonstration: Quickpropagation learning algorithm"<<std::endl;
    std::cout<<"Institute: BUT FIT"<<std::endl;
    std::cout<<"Course: Soft Computing"<<std::endl;
    std::cout<<"Author: Martin Fajcik xfajci00@stud.fit.vutbr.cz"<<std::endl;
    std::cout<<"Release Date: 05.12.2015"<<std::endl;
}

void warning(unsigned number)
{
    std::cout<<"WARNING: ";
    switch (number)
    {
    case (UNKNOWN_CONFIG_PARAMETER):
        std::cout<<"Unknown parameter encountered in configuration file\n";
        break;
    case (FILE_OPEN_FAILED):
        std::cout<<"Failed to open file. Returning to main menu.\n";
        break;
    case (INVALID_NEURONCOUNT_VALUE):
        std::cout<<"Encountered invalid layers_inputs_neurons configuration parameter value. Please check your configuration file. Returning to main menu.\n";
        break;
    case (IMPROPER_CONFIGURATION):
        std::cout<<"Loaded configuration is invalid, please check your configuration file\n";
        break;
    case (CONFIG_NOT_LOADED):
        std::cout<<"Configuration was not loaded yet. Returning to main menu.\n";
        break;
    case (WRONG_CONFIG_FORMAT):
        std::cout<<"Configuration file given has an incorrect formatting.\n";
        break;
    case (WRONG_INPUT_SIZE):
        std::cout<<"Number of inputs in data file differ from network number of inputs.\n";
        break;
    case (NN_NOTFOUND):
        std::cout<<"Network did not undergo any learning so far so it does not exists. Please train your network at least once at first. Returning to main menu.\n";
        break;
    }
}
bool read_configuration(Config* config)
{
    std::cout << "Please insert you configuration file name.\n";
    using namespace std;
    string filename;
    cin>>filename;
    ifstream config_file(filename);
    if (!config_file.is_open())
    {
        warning(FILE_OPEN_FAILED);
        return false;
    }
    string connection;
    bool mi = false, lambda = false, epsilon = false,maxiterations = false, threshold = false, layers_inputs_neurons = false;
    while (getline(config_file, connection))
    {
        istringstream ss(connection);

        string name;
        ss >> name;
        if (name == "mi")
        {
            ss>>config->mi;
            mi = true;
        }
        else if (name == "lambda")
        {
            ss>>config->lambda;
            lambda = true;
        }
        else if (name == "maxiterations")
        {
            ss>>config->maxiterations;
            maxiterations = true;
        }
        else if (name == "momentum")
        {
            ss>>config->momentum;
        }
        else if (name == "threshold")
        {
            ss>>config->threshold;
            threshold = true;
        }
        else if (name == "epsilon")
        {
            ss>>config->epsilon;
            epsilon = true;
        }
        else if (name == "layers_inputs_neurons")
        {
            ss>>config->layercount>>config->inputs;
            config->neurons = new unsigned[config->layercount+1];
            if (config->inputs<=0 || config->layercount<=0)
            {
                warning(INVALID_NEURONCOUNT_VALUE);
                return false;
            }
            config->neurons[0] = config->inputs;
            for (unsigned i = 1; i<=config->layercount; i++)
            {
                ss>>config->neurons[i];
                if (config->neurons[i]<=0)
                {
                    warning(INVALID_NEURONCOUNT_VALUE);
                    return false;
                }
            }
            layers_inputs_neurons = true;
        }
        else
            warning(UNKNOWN_CONFIG_PARAMETER);

    }
    configloaded = mi && threshold && lambda && layers_inputs_neurons && maxiterations && epsilon;
    if (!configloaded)
        warning(IMPROPER_CONFIGURATION);
    else
        std::cout<<"Configuration loaded successfully. Returning to main menu.\n";
    return configloaded;
}
bool read_datafile(double*** data, unsigned * matrix_height, unsigned * matrix_width)
{

    using namespace std;
    cout << "Please insert you data file name.\n";
    string filename;
    cin>>filename;
    ifstream file(filename);
    if (!file.is_open())
    {
        warning(FILE_OPEN_FAILED);
        return false;
    }
    string connection;

    if (getline(file, connection))
    {
        istringstream ss(connection);
        ss >> (*matrix_height) >> (*matrix_width);
        (*data) = new double* [(*matrix_height)];
        for (unsigned i = 0; i< (*matrix_height); i++)
            (*data)[i] = new double [(*matrix_width)];

    }
    else
    {
        warning(WRONG_INPUT_SIZE);
        return false;
    }
    for (unsigned lineindex=0; lineindex<(*matrix_height); lineindex++)
    {
        getline(file, connection);
        istringstream ss(connection);
        for (unsigned i =0; i < (*matrix_width); i++ )
            ss>>(*data)[lineindex][i];
    }
    return true;
}
void printHelp()
{
    std::cout<< "\nLegend:\n"
             <<"press C to read configuration from file\n"
             <<"press Q to print neuron network info (weights and configuration)\n"
             <<"press T to train network\n"
             <<"press R to restart network\n"
             <<"press O to test network\n"
             <<"press V to turn visualization on/off\n"
             <<"press H to show this help\n"
             <<"press I to show project information\n"
             <<"press E to exit application\n"
             <<"press D to activate/deactivate demonstrative training with\n\t backpropagation and quickpropagation\n"
             <<"\n";
}

void QPNetwork::summarize_information()
{
    using namespace std;
    cout<<"--Constants--\n";
    cout<<"mi ="<<this->_mi<<endl;
    cout<<"lambda ="<<this->_lambda<<endl;
    cout<<"momentum ="<<this->_momentum<<endl;
    cout<<"shrink factor ="<<this->_shrink<<endl;

    cout<<"\n--Network setup--\n";
    cout<<"Number of layers: "<<this->L<<endl;
    cout<<"Number of inputs: "<<this->n[0]<<endl;
    cout<<"Number of neurons for each layer: \n";
    for (unsigned i = 1; i<=this->L; i++)
        cout<<"Layer #"<<i<<" :"<<n[i]<<endl;

    cout<<"\nPress W to print weights or press any letter to return."<<endl;
    char option = 0;
    option = readoption();
    if (option != 'w' && option != 'W')
    {
        cout<<"\nReturning to main menu."<<endl;
        return;
    }
    cout<<"\n--Weights--\n";
    for (unsigned l = 1; l<=L; l++)
    {
        cout<<"Layer#"<<l<<": \n";
        for(unsigned j = 1; j<=n[l]; j++)
        {
            cout<<"Neuron#"<<j<<": \n";
            for (unsigned i=0; i<=n[l-1]; i++)
                std::cout<<"  Weight on input#"<<i<<": "<<w[l][j][i]<<std::endl;
            std::cout<<std::endl;
        }
    }

}
void QPNetwork::restart(Config* config)
{
    this->deallocate_network();
    this->setconfig(config);
    this->allocate_network();
}

void QPNetwork::setconfig(Config* config)
{
    _mi = config->mi;
    _lambda = config->lambda;
    _epsilon = config->epsilon;
    _momentum = config->momentum;
    _shrink = _epsilon/(1.0+_epsilon);

    L = config->layercount;
    n = config->neurons;
    maxiterations = config->maxiterations;
    threshold = config->threshold;
}

void QPNetwork::printlearningstatistics()
{
    std::cout << "\n\n----------------Learning finished---------------\n";
    std::cout << "GLerr: " << GLerr << std::endl;
    std::cout << "Iterations: " << iter << std::endl;
    if (!demo_bpqp)
        std::cout << "Quick propagation steps: " << QPJ_count<<std::endl;
    std::cout << "------------------------------------------------\n";
    QPJ_count=QPJ_count^QPJ_count;
}
void QPNetwork::validate_learning()
{
    if (GLerr<threshold)
        std::cout<<"Learning SUCCESSFUL, returning to main menu.\n";
    else
        std::cout<<"Learning FAILED, algorithm probably got stuck in local minimum, try changing initial configuration\n";
}
void precalculate(scene* s,QPNetwork* qp_nn)
{

    //calculate circles and text inside
    unsigned width = NN_WIDTH/qp_nn->L;
    cairo_t *cr;
    cairo_surface_t *cairosurf;

    cairosurf = cairo_image_surface_create_for_data(
                    (unsigned char*)screen->pixels,
                    CAIRO_FORMAT_ARGB32,
                    screen->w,
                    screen->h,
                    (Uint32)screen->pitch);

    cr = cairo_create(cairosurf);

    //cairo_text_extents_t te;
    neuron* n;
    connection* c;
    unsigned x = 100;
    for (unsigned i=0; i<=qp_nn->L; i++)
    {

        unsigned height=  NN_HEIGHT/qp_nn->n[i];
        unsigned diameter = (width<height ? width:height)/2;
        if (diameter>MAX_NEURON_DIAMETER)
            diameter = MAX_NEURON_DIAMETER;

        unsigned y =100;
        for (unsigned j =(i == qp_nn->L?1:0); j<=qp_nn->n[i]; j++ )
        {
            unsigned local_diameter = (i==0 || j == 0 ? diameter/2:diameter);

            n = new neuron;
            s->neurons.push_back(n);
            n->p.x = x;
            n->p.y = y;
            n->radius = local_diameter;
            //double fontSize = MAX_FONT_SIZE;
            n->nnpos.x = i;
            n->nnpos.y = j;
            if (i>0&& j>0 && i<=qp_nn->L)
            {
                int z = 0;
                for (neuron* neuron: s->neurons)
                {
                    if (neuron->nnpos.x == i-1)
                    {
                        c = new connection;
                        c->p2.x = x-local_diameter;
                        c->p2.y = y;
                        c->p1.x = neuron->p.x+neuron->radius;
                        c->p1.y = neuron->p.y;						
						c->angle = atan((double)(c->p2.y-c->p1.y)/(double)(c->p2.x-c->p1.x));
                        c->x = n->nnpos.x;
                        c->y = n->nnpos.y;
                        c->z = z;
                        z++;
                        s->connections.push_back(c);
                    }
                }
            }
            if (i==qp_nn->L)
            {
                c = new connection;
                c->p1.x = x+local_diameter;
                c->p1.y = y;
                c->p2.x = c->p1.x+100;
                c->p2.y = y;
                c->x = std::numeric_limits<unsigned>().max();
                c->y = std::numeric_limits<unsigned>().max();
                c->z = std::numeric_limits<unsigned>().max();
                s->connections.push_back(c);
            }
            y+=height;
        }
        x+=width;
    }
    cairo_destroy(cr);
}
void run_visdemo(QPNetwork* qp_nn, unsigned matrix_height, double** data)
{
    window_closed = true;
    std::thread uithread(display_algorithm,qp_nn);
    qp_nn->learn(data, matrix_height, false);
    qp_nn->printlearningstatistics();
    qp_nn->validate_learning();
    if (!window_closed)
        std::cout<<"If you want to continue, please close the display window\n";
    uithread.join();
}
void run_bpqpdemo(QPNetwork* qp_nn, Config config, unsigned matrix_height, double** data)
{
    double*** wtmp = new double**[qp_nn->L + 1]();
    for (unsigned l = 0; l < qp_nn->L + 1; l++)
    {
        wtmp[l] = new double*[qp_nn->n[l] + 1]();
        for (unsigned j = 0; j < qp_nn->n[l] + 1; j++)
        {
            if (l > 0)
            {
                wtmp[l][j] = new double[qp_nn->n[l - 1] + 1]();
                for (unsigned i = 0; i < qp_nn->n[l - 1] + 1; i++)
                    wtmp[l][j][i] = qp_nn->w[l][j][i];
            }
            else
            {
                wtmp[l][j] = new double[qp_nn->n[0] + 1]();
                for (unsigned i = 0; i < qp_nn->n[0] + 1; i++)
                    wtmp[l][j][i] =  qp_nn->w[l][j][i];
            }
        }
    }

    qp_nn->learn(data, matrix_height, demo_bpqp);
    double bp_GLerr = qp_nn->GLerr;
    double bp_iter = qp_nn->iter;
    qp_nn->deallocate_network();
    qp_nn->allocate_network();
    for (unsigned l = 0; l < qp_nn->L + 1; l++)
    {
        for (unsigned j = 0; j < qp_nn->n[l] + 1; j++)
            delete[] qp_nn->w[l][j];
        delete[] qp_nn->w[l];
    }
    delete[] qp_nn->w;
    qp_nn->w = wtmp;
    qp_nn->learn(data, matrix_height, !demo_bpqp);
    std::cout << "Backprop iterations: "<<bp_iter << "\nQuickprop iterations: "<<qp_nn->iter<<std::endl;
    if (qp_nn->GLerr<config.threshold && bp_GLerr<config.threshold)
        std::cout<<"Learning successful, returning to main menu.\n";
    else
        std::cout<<"Learning unsuccessful :(, algorithm probably got stuck in local minimum, try changning the network configuration?\n";
}
char readoption()
{
    char option = 0;
    while (option <'A')
        option = getchar();
    return option;
}

void run_program_loop()
{
    printInfo();
    printHelp();
    bool qp_nn_created = false;
    char option = 0;
    QPNetwork* qp_nn = nullptr;
	Config config = {0.0,0.0,0.0,0.0,0.0,0,0,0,nullptr},oldConfig;
    while (option != 'e' || option != 'E' )
    {
        option = readoption();
        switch (option)
        {
        case 'e':
        case 'E':
            goto exitLoop;
            break;
        case 'v':
        case 'V':
            visualize=!visualize;
            std::cout<<"\nNetwork visualization has been turned "<<(visualize ? "on":"off")<<std::endl;
            if  (demo_bpqp)
            {
                std::cout<<"Turning off demonstrative training with backpropagation and quickpropagation\n";
                demo_bpqp = false;
            }
            break;
        case 'c':
        case 'C':
			{
				if (configloaded)
					config.copy(&oldConfig);
				bool success = read_configuration(&config);
				if (success && qp_nn_created)
				{
					if (config.inputs != qp_nn->n[0] ||
							config.layercount != qp_nn->L)
					{
						std::cout<< "By modifying network size all learned data will be erased. Do you wish to proceed? (Y/N):\n";
						option = readoption();
						if (option == 'y' || option=='Y' )
						{
							qp_nn->restart(&config);
							std::cout<< "The network configuration has been changed.\n";
						}
						else
						{
							config = oldConfig;
							std::cout<< "Keeping old configuration and returning to main menu.\n";
						}
					}
					else
					{
						for (unsigned i = 1; i<=qp_nn->L; i++)
						{
							if (config.neurons[i]!=qp_nn->n[i])
							{
								std::cout<< "By modifying network size all learned data will be erased. Do you wish to proceed? (Y/N):\n";
								option = readoption();
								if (option == 'y' || option=='Y' )
								{
									qp_nn->restart(&config);
									std::cout<< "The network configuration has been changed.\n";
								}
								else
								{
									config = oldConfig;
									std::cout<< "Keeping old configuration and returning to main menu.\n";
									break;
								}
							}
						}
					}
				}
			}
            break;
        case 'h':
        case 'H':
            printHelp();
            break;
        case 'i':
        case 'I':
            printInfo();
            break;
        case 'q':
        case 'Q':
            if (!configloaded)
            {
                warning(CONFIG_NOT_LOADED);
                continue;
            }
            else if (!qp_nn_created)
            {
                warning(NN_NOTFOUND);
                continue;
            }
            else qp_nn->summarize_information();
            break;
        case 'd':
        case 'D':
            demo_bpqp = !demo_bpqp;
            std::cout<<"\nQuick propagation vs Back propagation demo has been "<<(demo_bpqp ? "activated":"deactivated")<<std::endl;
            if  (visualize)
            {
                std::cout<<"Turning off network visualization\n";
                visualize = false;
            }
            break;
        case 't':
        case 'T':
        {
            if (!configloaded)
            {
                warning(CONFIG_NOT_LOADED);
                continue;
            }
            std::cout<<"\npress L to load data from file\n";
            option = readoption();

            double** data;
            unsigned matrix_height = 0,matrix_width = 0;
            if (option == 'l' || option == 'L')
            {
                if (!read_datafile(&data, &matrix_height, &matrix_width))
                    continue;
            }
            else
            {
                std::cout<<"Invalid option '"<<option<<"'. Returning to main menu.\n";
                continue;

            }

            if (!qp_nn_created)
            {
                qp_nn = new QPNetwork(&config);
                qp_nn_created = true;
            }
            if (matrix_width != config.inputs+qp_nn->n[qp_nn->L])
                warning(WRONG_INPUT_SIZE);
            else
            {
                if(demo_bpqp)
                    run_bpqpdemo(qp_nn,config, matrix_height,  data);
                else if (visualize)
                    run_visdemo(qp_nn, matrix_height,  data);
                else
                {
                    qp_nn->learn(data, matrix_height, false);
                    qp_nn->printlearningstatistics();
                    qp_nn->validate_learning();
                }
            }
            for (unsigned i =0; i<matrix_height; i++)
                delete[] data[i];
            delete[] data;
        }
        break;
        case 'r':
        case 'R':
        {
            if (!configloaded)
            {
                warning(CONFIG_NOT_LOADED);
                continue;
            }
            else if (!qp_nn_created)
            {
                warning(NN_NOTFOUND);
                continue;
            }
            else
            {
                qp_nn->restart(&config);
                std::cout<< "The network has been restarted. Returning to main menu.\n";
            }
        }
        break;
        case 'o':
        case 'O':
        {
            if (!configloaded)
            {
                warning(CONFIG_NOT_LOADED);
                continue;
            }
            else if (!qp_nn_created)
            {
                warning(NN_NOTFOUND);
                continue;
            }
            std::cout<<"\npress L to load data from file.\n";
            double** data;
            unsigned matrix_height = 0,matrix_width = 0;
            option = readoption();
            if (option == 'l' || option == 'L')
            {
                if (!read_datafile(&data, &matrix_height, &matrix_width))
                    continue;
            }
            else
            {
                std::cout<<"Invalid option '"<<option<<"'. Returning to main menu.";
                continue;
            }
            if (matrix_width != config.inputs)
                warning(WRONG_INPUT_SIZE);
            else if (qp_nn!= nullptr)
                qp_nn->test(data, matrix_height);
            for (unsigned i =0; i<matrix_height; i++)
                delete[] data[i];
            delete[] data;
        }
        break;
        default:
            std::cout<<"Invalid option "<< option << ". Please re-enter your choice.\n";
            break;
        }
    }
exitLoop:
    if (qp_nn_created)
        delete qp_nn;
}

int main(int argc, char* argv[])
{
    run_program_loop();
    return 0;
}