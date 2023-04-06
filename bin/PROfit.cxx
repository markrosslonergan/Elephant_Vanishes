#include "PROconfig.h"
#include "PROspec.h"
#include "PROcovariancegen.h"

#include "CLI11.h"
#include "LBFGSB.h"

#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/NumericalDiff>

#include "TH1D.h"

#include <chrono>
#include <random>


using namespace PROfit;
log_level_t GLOBAL_LEVEL = LOG_DEBUG;

const std::vector<double> linspace(double start, double end, int num_points) {
    std::vector<double> result(num_points);
    double delta = (end - start) / (num_points - 1);
    std::iota(result.begin(), result.end(), start);
    for (int i = 1; i < num_points; i++) {
        result[i] = start + i * delta;
    }
    return result;
}

std::vector<std::pair<double, int>> fill_rand(int numVals, int numSC, double low, double high) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<> distS{0,numSC-1};
    std::normal_distribution<> distV{(high+low)/2, (high-low)/4};
    std::vector<std::pair<double, int>> ret;
    ret.reserve(numVals);
    for(int i = 0; i < numVals; ++i) {
        ret.push_back(std::make_pair(distV(rng), distS(rng)));
    }
    return ret;
}

int FindLocalBin(double reco_value, std::vector<double>& bins, std::vector<double>::iterator& bins_begin, std::vector<double>::iterator& bins_end){

    auto it = std::lower_bound(bins_begin, bins_end, reco_value);
    if (it == bins_end) {
        return static_cast<int>(bins.size());
    }
    return static_cast<int>(std::distance(bins_begin, it));
}


int FindLocalBin2(double reco_value, std::vector<double>& bin_edges){
    auto pos_iter = std::upper_bound(bin_edges.begin(), bin_edges.end(), reco_value);
    if(pos_iter == bin_edges.end() || pos_iter == bin_edges.begin()){
        return 0; 
    }
    return std::distance(bin_edges.begin(),pos_iter) - 1;
}
std::vector<double> hist_test(const std::vector<std::pair<double, int>>& vals, std::vector<double>& bin_edges, int numSC) {
    std::vector<double> hists(numSC * (bin_edges.size()-1), 0);

    auto start = std::chrono::steady_clock::now();
    for(const auto&[value, subCH] : vals) {
        if(value < bin_edges[0]) continue;
        for(int i = 1; i < bin_edges.size(); ++i) {
            if(bin_edges[i] > value) {
                hists[(bin_edges.size() - 1) * subCH + i-1] += 1;
                break;
            }
        }
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time elapsed: " << diff.count() << std::endl;
    return hists;
}


class ChiTest
{
    private:
        int n;
    public:
        ChiTest(int n_) : n(n_) {}
        double operator()(const Eigen::VectorXd &x, Eigen::VectorXd &grad)
        {
            double fx = 0.0;
            for(int i = 0; i < n; i += 2)
            {
                double t1 = 1.0 - x[i];
                double t2 = 10 * (x[i + 1] - x[i] * x[i]);
                grad[i + 1] = 20 * t2;
                grad[i]     = -2.0 * (x[i] * grad[i + 1] + t1);
                fx += t1 * t1 + t2 * t2;
            }
            return fx;
        }
};


int main(int argc, char* argv[])
{

    std::cout<<"PROfit"<<std::endl;
    CLI::App app{"Test for PROfit"}; 

    // Define options
    std::string xmlname = "NULL.xml"; 
    int maxevents = 100;

    int which = 0;

    //doubles
    app.add_option("-x,--xml", xmlname, "Input PROfit XML config.");
    app.add_option("-m,--max", maxevents, "Max number of events to run over.");
    app.add_option("-w,--which", which, "Max number of events to run over.");
    app.add_option("-v,--verbosity", GLOBAL_LEVEL, "Verbosity Level [1-4].");

    CLI11_PARSE(app, argc, argv);


    std::cout<<"PROfit"<<__LINE__<<std::endl;

    //PROconfig myConf(xmlname);
    //PROspec mySpec(myConf);
    //TH1D hmm = mySpec.toTH1D(myConf);




    int num_sc = 10;
    std::vector<int> nn={10000000}; //, 3000000, 10000000, 30000000, 100000000};
std::vector<int> nb={100, 1000};

for (int n : nn){
    for (int nnb: nb){
        std::vector<double> bin_edges  = linspace(0,10, nnb);

        std::cout<<"Using "<<nnb<<" bins and "<<n<<" events"<<std::endl;
        std::vector<double> ran;
        std::vector<int> rani;

        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<> dist1{0,10};
        std::uniform_int_distribution<> dist2{0,num_sc-1};


        for(int i=0; i<n; i++){
            ran.push_back(dist1(rng));
            rani.push_back(dist2(rng));
        }

        if(which==0){
            std::cout<<" Jacob vector"<<std::endl;
            auto start0 = std::chrono::high_resolution_clock::now();
            ///////////////////////////////////////////////////
            std::vector<double> hists(num_sc * (bin_edges.size()-1), 0);
            /////////////////////////////////////////////////
            auto end00 = std::chrono::high_resolution_clock::now();
            auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(end00 - start0);
            std::cout << "Time taken by setup: " << duration0.count() << " microseconds per fill" << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            ////////////////////////////////////////////////
            for(int j=0; j<n; j++){
                if(ran[j] < bin_edges[0]) continue;
                for(int i = 1; i < bin_edges.size(); ++i) {
                    if(bin_edges[i] > ran[j]) {
                        hists[(bin_edges.size() - 1) * rani[j] + i-1] += 1;
                        break;
                    }
                }
            } 
            ///////////////////////////////////////////////
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            std::cout << "Time taken by function: " << duration.count()/(double)n << " nanoseconds per fill" << std::endl;

        }




        if(which==1){
            std::cout<<" BASIC ROOT "<<std::endl;
            auto start0 = std::chrono::high_resolution_clock::now();
            ///////////////////////////////////////////////////
            std::vector<TH1D*> hists;
            for(int i=0; i< num_sc; i++){
                TH1D *h = new TH1D(std::to_string(i).c_str(),std::to_string(i).c_str(),bin_edges.size()-1,&bin_edges[0]);
                hists.push_back(h);
            }
            /////////////////////////////////////////////////
            auto end00 = std::chrono::high_resolution_clock::now();
            auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(end00 - start0);
            std::cout << "Time taken by setup: " << duration0.count() << " microseconds per fill" << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            ////////////////////////////////////////////////
            for(int i=0; i< n; i++){
                hists[rani[i]]->Fill(ran[i]);
            }
            ///////////////////////////////////////////////
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            std::cout << "Time taken by function: " << duration.count()/(double)n << " nanoseconds per fill" << std::endl;
        }

        if(which==4){
            std::cout<<" Guanqun Coded"<<std::endl;

            auto start0 = std::chrono::high_resolution_clock::now();
            ///////////////////////////////////////////////////
            std::vector<double>::iterator bins_begin = bin_edges.begin();
            std::vector<double>::iterator bins_end = bin_edges.end();
            std::vector<double> hists(num_sc * (bin_edges.size()), 0);
            std::vector<int> startp;
            for(int k=0; k<num_sc; k++){
                startp.push_back((k==0 ? 0 : startp.back())+bin_edges.size()-1);
            }

            /////////////////////////////////////////////////
            auto end00 = std::chrono::high_resolution_clock::now();
            auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(end00 - start0);
            std::cout << "Time taken by setup: " << duration0.count() << " microseconds per fill" << std::endl;
            auto start = std::chrono::high_resolution_clock::now();

            ////////////////////////////////////////////////
            for(int j=0; j<n; j++){
                hists[startp[rani[j]]+FindLocalBin2(ran[j],bin_edges)]+=1;
            } 

            ///////////////////////////////////////////////
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            std::cout << "Time taken by function: " << duration.count()/(double)n << " nanoseconds per fill" << std::endl;

        }




        if(which==2){
            std::cout<<" Mark Eigen"<<std::endl;

            auto start0 = std::chrono::high_resolution_clock::now();
            ///////////////////////////////////////////////////
            Eigen::VectorXd hists = Eigen::VectorXd::Zero(num_sc * (bin_edges.size()));
            std::vector<int> startp;
            auto bins_begin = bin_edges.begin();
            auto bins_end = bin_edges.end();

            for(int k=0; k<num_sc; k++){
                startp.push_back((k==0 ? 0 : startp.back())+bin_edges.size()-1);
            }
            /////////////////////////////////////////////////
            auto end00 = std::chrono::high_resolution_clock::now();
            auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(end00 - start0);
            std::cout << "Time taken by setup: " << duration0.count() << " microseconds per fill" << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            ////////////////////////////////////////////////
            for(int j=0; j<n; j++){
                hists[startp[rani[j]]+FindLocalBin(ran[j],bin_edges,bins_begin,bins_end)]+=1;
            } 
            ///////////////////////////////////////////////
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            std::cout << "Time taken by function: " << duration.count()/(double)n << " nanoseconds per fill" << std::endl;

        }




        if(which==3){
            std::cout<<" Ibrahim Chat"<<std::endl;

            auto start0 = std::chrono::high_resolution_clock::now();
            ///////////////////////////////////////////////////
            std::vector<double>::iterator bins_begin = bin_edges.begin();
            std::vector<double>::iterator bins_end = bin_edges.end();
            std::vector<double> hists(num_sc * (bin_edges.size()), 0);
            std::vector<int> startp;
            for(int k=0; k<num_sc; k++){
                startp.push_back((k==0 ? 0 : startp.back())+bin_edges.size()-1);
            }

            /////////////////////////////////////////////////
            auto end00 = std::chrono::high_resolution_clock::now();
            auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(end00 - start0);
            std::cout << "Time taken by setup: " << duration0.count() << " microseconds per fill" << std::endl;
            auto start = std::chrono::high_resolution_clock::now();

            ////////////////////////////////////////////////
            for(int j=0; j<n; j++){
                hists[startp[rani[j]]+FindLocalBin(ran[j],bin_edges,bins_begin,bins_end)]+=1;
            } 

            ///////////////////////////////////////////////
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            std::cout << "Time taken by function: " << duration.count()/(double)n << " nanoseconds per fill" << std::endl;

        }










        std::cout<<"PROfit"<<__LINE__<<std::endl;



        /*
           LBFGSpp::LBFGSBParam<double> param;  
           param.epsilon = 1e-6;
           param.max_iterations = 100;
           LBFGSpp::LBFGSBSolver<double> solver(param); 

           int n=78;
           ChiTest fun(n);

        // Bounds
        Eigen::VectorXd lb = Eigen::VectorXd::Constant(n, 0.0);
        Eigen::VectorXd ub = Eigen::VectorXd::Constant(n, std::numeric_limits<double>::infinity());

        // Initial guess
        Eigen::VectorXd x = Eigen::VectorXd::Constant(n, 2.0);


        // x will be overwritten to be the best point found
        double fx;
        int niter = solver.minimize(fun, x, fx, lb, ub);


        std::cout << niter << " iterations" << std::endl;
        std::cout << "x = \n" << x.transpose() << std::endl;
        std::cout << "f(x) = " << fx << std::endl;
        */
    }
}
return 0;
}

