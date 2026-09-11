#include "PROtocall.h"
#include "PROlog.h"
#include <atomic>

namespace PROfit{


    int FindLocalVariableBin(const PROconfig &inconfig, const BranchVariable::Value &other_value, int channel_index, int other_index) {
        //find local bin 
        const PROconfig::Binning &bins = inconfig.GetChannelVariableBins(channel_index, other_index);
        int ret = bins.Bin(other_value);
        if (ret == -1) {
            log<LOG_DEBUG>(L"%1% || Channel index : %2% Variable index : %3% returned under or overflow") % __func__ % channel_index % other_index;
        }
        return ret;
    }

    int FindGlobalVariableBin(const PROconfig &inconfig, const BranchVariable::Value &other_value, int subchannel_index, int other_index) {
        int global_bin_start = inconfig.GetGlobalVariableBinStart(subchannel_index, other_index);
        int channel_index = inconfig.GetLocalChannelIndexFromGlobalSubchannelIndex(subchannel_index);
        if (inconfig.GetChannelVariableBins(channel_index, other_index).NBins() == 0) {
            log<LOG_ERROR>(L"%1% || Subchannel %2% does not have other bins") % __func__ % subchannel_index;
            log<LOG_ERROR>(L"%1% || Return global bin of -1") % __func__ ;
            return -1;
        }
        int local_bin = FindLocalVariableBin(inconfig, other_value, channel_index, other_index);
        return local_bin == -1 ? -1 : global_bin_start + local_bin;
    }


    int FindGlobalVariableBin(const PROconfig &inconfig, const BranchVariable::Value &other_value, const std::string& subchannel_fullname, int other_index) {
        int subchannel_index = inconfig.GetSubchannelIndex(subchannel_fullname);
        return FindGlobalVariableBin(inconfig, other_value, subchannel_index, other_index);
    }


    Eigen::MatrixXf CollapseMatrix(const PROconfig &inconfig, const Eigen::MatrixXf& full_matrix){
        const Eigen::SparseMatrix<float>& T = inconfig.GetCollapsingMatrixSparse();
        const Eigen::Index num_bin_before_collapse = T.rows();
        if(full_matrix.rows() != num_bin_before_collapse || full_matrix.cols() != num_bin_before_collapse){
            log<LOG_ERROR>(L"%1% || Matrix dimension doesn't match expected size. Provided matrix: %2% x %3%. Expected matrix size: %4% x %5%") % __func__ % full_matrix.rows() % full_matrix.cols() % num_bin_before_collapse% num_bin_before_collapse;
            log<LOG_ERROR>(L"Terminating.");
            exit(EXIT_FAILURE);
        }
        return Eigen::MatrixXf(T.transpose() * full_matrix * T);
    }

    Eigen::VectorXf CollapseMatrix(const PROconfig &inconfig, const Eigen::VectorXf& full_vector){
        const Eigen::SparseMatrix<float>& T = inconfig.GetCollapsingMatrixSparse();
        if(full_vector.size() != T.rows()){
            log<LOG_ERROR>(L"%1% || Vector dimension doesn't match expected size. Provided vector size: %2% . Expected size: %3%") % __func__ % full_vector.size() % T.rows();
            log<LOG_ERROR>(L"Terminating.");
            exit(EXIT_FAILURE);
        }
        return Eigen::VectorXf(T.transpose() * full_vector);
    }

    Eigen::MatrixXf CollapsedScaledCovariance(const PROconfig &inconfig, const Eigen::MatrixXf& frac_cov, const Eigen::VectorXf& spec){
        const Eigen::SparseMatrix<float>& T = inconfig.GetCollapsingMatrixSparse();
        if(frac_cov.rows() != T.rows() || frac_cov.cols() != T.rows() || spec.size() != T.rows()){
            log<LOG_ERROR>(L"%1% || Dimension mismatch: frac_cov %2%x%3%, spec %4%, expected %5%.")
                % __func__ % frac_cov.rows() % frac_cov.cols() % spec.size() % T.rows();
            log<LOG_ERROR>(L"Terminating.");
            exit(EXIT_FAILURE);
        }
        // S = diag(spec) * T stays sparse (same pattern as T, rows scaled).
        Eigen::SparseMatrix<float> S = spec.asDiagonal() * T;
        Eigen::MatrixXf FS = frac_cov * S;               // dense (N x m), no N x N temporary
        return Eigen::MatrixXf(S.transpose() * FS);      // dense (m x m)
    }

    Eigen::VectorXf ChannelNormFactors(const PROconfig &inconfig, const Eigen::VectorXf &collapsed_pred, const Eigen::VectorXf &data, int other_index){
        if(collapsed_pred.size() != data.size()){
            log<LOG_ERROR>(L"%1% || Size mismatch: collapsed prediction %2% vs data %3%.") % __func__ % collapsed_pred.size() % data.size();
            log<LOG_ERROR>(L"Terminating.");
            exit(EXIT_FAILURE);
        }
        Eigen::VectorXf r = Eigen::VectorXf::Ones(collapsed_pred.size());
        size_t global_channel_index = 0;
        for(size_t im = 0; im < inconfig.m_num_modes; ++im)
            for(size_t id = 0; id < inconfig.m_num_detectors; ++id)
                for(size_t ic = 0; ic < inconfig.m_num_channels; ++ic){
                    const size_t nbin  = inconfig.GetChannelVariableBins(global_channel_index, other_index).NBins();
                    const size_t start = inconfig.GetCollapsedGlobalVariableBinStart(global_channel_index, other_index);
                    const float sp = collapsed_pred.segment(start, nbin).sum();
                    const float sd = data.segment(start, nbin).sum();
                    if(sp > 0.0f && std::isfinite(sp) && std::isfinite(sd)) {
                        r.segment(start, nbin).setConstant(sd / sp);
                    } else {
                        static std::atomic<bool> warned{false};
                        if(!warned.exchange(true))
                            log<LOG_WARNING>(L"%1% || Shape-only: channel %2% has prediction integral %3%; its normalisation factor is left at 1 (warned once).") % __func__ % global_channel_index % sp;
                    }
                    ++global_channel_index;
                }
        return r;
    }

    Eigen::VectorXf ShapeRescaleToData(const PROconfig &inconfig, const Eigen::VectorXf &full_spec, const Eigen::VectorXf &data, int other_index, Eigen::VectorXf *r_out){
        const Eigen::SparseMatrix<float>& T = inconfig.GetCollapsingMatrixSparse(other_index);
        if(full_spec.size() != T.rows()){
            log<LOG_ERROR>(L"%1% || Spectrum size %2% does not match the uncollapsed binning (%3%) of variable %4%.") % __func__ % full_spec.size() % T.rows() % other_index;
            log<LOG_ERROR>(L"Terminating.");
            exit(EXIT_FAILURE);
        }
        const Eigen::VectorXf collapsed = T.transpose() * full_spec;
        const Eigen::VectorXf r = ChannelNormFactors(inconfig, collapsed, data, other_index);
        if(r_out) *r_out = r;
        return full_spec.cwiseProduct(Eigen::VectorXf(T * r));
    }

    Eigen::MatrixXf CollapseMatrix(const PROconfig &inconfig, const Eigen::MatrixXf& full_matrix, int other_index){
        const Eigen::SparseMatrix<float>& T = inconfig.GetCollapsingMatrixSparse(other_index);
        const Eigen::Index num_bin_before_collapse = T.rows();
        if(full_matrix.rows() != num_bin_before_collapse || full_matrix.cols() != num_bin_before_collapse){
            log<LOG_ERROR>(L"%1% || Matrix dimension doesn't match expected size. Provided matrix: %2% x %3%. Expected matrix size: %4% x %5%") % __func__ % full_matrix.rows() % full_matrix.cols() % num_bin_before_collapse% num_bin_before_collapse;
            log<LOG_ERROR>(L"Terminating.");
            exit(EXIT_FAILURE);
        }
        return Eigen::MatrixXf(T.transpose() * full_matrix * T);
    }

    Eigen::VectorXf CollapseMatrix(const PROconfig &inconfig, const Eigen::VectorXf& full_vector, int other_index){
        const Eigen::SparseMatrix<float>& T = inconfig.GetCollapsingMatrixSparse(other_index);
        if(full_vector.size() != T.rows()){
            log<LOG_ERROR>(L"%1% || Vector dimension doesn't match expected size. Provided vector size: %2% . Expected size: %3%") % __func__ % full_vector.size() % T.rows();
            log<LOG_ERROR>(L"Terminating.");
            exit(EXIT_FAILURE);
        }
        return Eigen::VectorXf(T.transpose() * full_vector);
    }
    void PrintVariableInfo(const PROconfig &inconfig){

        int global_channel_index = 0;
        for(size_t im = 0; im < inconfig.m_num_modes; im++){
            for(size_t id =0; id < inconfig.m_num_detectors; id++){
                for(size_t ic = 0; ic < inconfig.m_num_channels; ic++){
                    for(size_t sc = 0; sc < inconfig.m_num_subchannels[ic]; sc++){

                        std::string temp_name  = inconfig.m_mode_names[im] +"_" +inconfig.m_detector_names[id]+"_"+inconfig.m_channel_names[ic]+"_"+inconfig.m_subchannel_names[ic][sc];
                        //global subchannel index is the main marker/identifier.
                        int global_subchannel_index1 = inconfig.GetSubchannelIndex(temp_name);
                        int local_channel_index1 = inconfig.GetLocalChannelIndexFromGlobalSubchannelIndex(global_subchannel_index1);
                        int local_channel_index3 = inconfig.GetLocalChannelIndexFromGlobalChannelIndex(global_channel_index);
                        int local_channel_index2 = ic;
                        int local_subchannel_index = sc;

                        log<LOG_INFO>(L"%1% || im:id %2%:%3%  %4%  || local_channel_index (%5%,%6%,%7%) global_channel_index (%8%) || local_subchannel_index %9% global (%10%)") % __func__ % im % id % temp_name.c_str() %  local_channel_index1 % local_channel_index2 % local_channel_index3 % global_channel_index % local_subchannel_index % global_subchannel_index1 ;

                        for(size_t io = 0; io < inconfig.m_num_variables; ++io) {

                            int nstart = inconfig.GetGlobalVariableBinStart(global_subchannel_index1,io);
                            int nbin = inconfig.GetChannelVariableBins(local_channel_index1,io).NBins();
                            std::vector<float> edges = inconfig.GetChannelVariableBins(local_channel_index1,io).Edges();
                            log<LOG_INFO>(L"%1% ||  ---  Vartiable %2%  || nstart %3% nbin %4% ") % __func__ % io %nstart % nbin ;
                            log<LOG_INFO>(L"%1% ||  --- -- Edges %2% ") % __func__ % edges ;
                        }
                    }
                global_channel_index++;
                }
            }
        }
        return;
    }


    
    Eigen::MatrixXf ComputeSquareRootCovariance(
            const Eigen::MatrixXf& covariance,
            bool use_ldlt, float svd_tol  ) {

        if (use_ldlt) {
            // --- LDLT decomposition (faster for symmetric matrices) ---
            // Note LDLT reports Success even for indefinite input; sqrt of a
            // negative D entry would silently produce NaNs (e.g. from a
            // rank-deficient sample covariance in MCMC proposal tuning), so
            // clamp negative entries to zero with a warning.
            Eigen::LDLT<Eigen::MatrixXf> ldlt(covariance);
            if (ldlt.info() != Eigen::Success) {
                throw std::runtime_error("LDLT decomposition failed.");
            }

            Eigen::VectorXf D = ldlt.vectorD();
            int n_clamped = 0;
            for (Eigen::Index i = 0; i < D.size(); ++i) {
                if (D(i) < 0) { D(i) = 0; ++n_clamped; }
            }
            if (n_clamped > 0) {
                log<LOG_WARNING>(L"%1% || Covariance is not positive semi-definite: clamped %2% negative LDLT pivot(s) to zero before taking the square root.")
                    % __func__ % n_clamped;
            }

            // L = P^T * L_p * D^{1/2}
             Eigen::MatrixXf Lp = ldlt.matrixL();
             Eigen::VectorXf D_sqrt = D.array().sqrt();
             Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P(ldlt.transpositionsP());
             return  P.transpose() * Lp * D_sqrt.asDiagonal();
        } else {

            Eigen::JacobiSVD<Eigen::MatrixXf> svd(covariance, Eigen::ComputeThinU | Eigen::ComputeThinV);
            const auto& U = svd.matrixU();
            const auto& S = svd.singularValues();

            // Zero singular values below the relative tolerance (svd_tol < 0
            // means keep everything, the historical behavior).
            Eigen::VectorXf S_kept = S;
            if (svd_tol > 0 && S.size() > 0) {
                const float tol = svd_tol * S.maxCoeff();
                for (Eigen::Index i = 0; i < S_kept.size(); ++i)
                    if (S_kept(i) < tol) S_kept(i) = 0;
            }
            Eigen::VectorXf S_sqrt = S_kept.array().sqrt();

            return U*S_sqrt.asDiagonal();
        }
    }


    std::string getIcon(){

        std::string icon = R"(
            ....                                                                                                  
            .=+=-..                                            .:                                                 
            .:=++=-.                              .-=+*****+=-+@=.   :+##-  :#%=                                 
            .-+++=-.               .::.      .*#*=--:::--+#@@@=  :#=*@%. :@@=                                 
            .:=+++=-.            :++=-.    ...         :**=-.  .. %@#. +@+                                  
            .:-++++=:.        .=++++=-.                        :@@= =@+                                   
            .-=++++=:.     -++++++++-.                      =@@=*%-                                    
            :=+-. .-+= .:=+++++=:. .+++++++++++-.             .--.   .*%#=.                                     
            .=+=%%:  +@*.   .-++++++==+++++++++++++-.           -+++:                                             
            .. :%%: .##.      .:=++++++++++:.-+++++++-.        :+++++-.                                           
            *@* .*#.      .:..:=+++++++-   .-+++++++-.     .+++++++=:                    .::.                  
            .#@=:*+.       :*=   .-++++=.     .-+++++++-.  .=+++++++++-.                  .+*#*-.               
            =##+-         :*=     .:==.        .-=++++++-.-+++++++++++=.                    .-*%=              
            .:=+. ..        .:.-*+:::.                .:=++++++++++++=+++++++:                      -@*.            
            :**%%*.          =*+++++++.       .=-.       .-++++++++++-.:=++++++-.                     :@+            
            ..*+:*:          =*+++++++.       :++.         .-=++++++=.  .-++++++=:                     +@:           
            -%.             =*+**++++.    .::-++:::.        .:+++++.     :+++++++-.                 .+=@==.         
            +*              .::::++++.    -++++++++:       .. .-=+:       .=++++++=.                .+@@@*.         
            +#          :======: -**+.    -++++++++:      .=+:  ..         .:+++++++-.                :=:           
            :%:        .-******- .-=+.    -++++++++:       +*:               .=++++++=.                :-.   .-:.   
            +#.   .:-==++++++++==-:.     -++++++++:   .---++=--:      .::.   .-+++++++:            .-##@@:  *@@.   
            .-: :=+*****+++++++***++-.   -++++++++:   :***++***-      .++:     :=++++++-.          .=::@@-  *@+.   
            .=+*+++*+++++++++*++++*+:  -++++++++:   :+++++++*-   ....=+:...   .-++++++=:     ...    +@@. =@+     
            .=*+++++-..:*++++.-+*+++*+. -++++++++:   :+++++++*-   .===+++=+:     :+++++++-.  .=+:   .%@*.+@=      
            .+++++++.  :*++++. .+++++=: -++++++++:   :+++++++*-   .++++++++:      .=++++++=..=++=.  .#@%##:       
            .+++++++=..:*++++.  .....   -++++++++:   :+++++++*-   .++++++++:       .:+++++++=++++.   .-=:         
            .-*+++++*++++++++.          -++++++++:   :+++++++*-   .++++++++:     ....:+++++++++++-                
            .-+**++++++++++++==-:.     -++++++++:   :+++++++*-   .++++++++:  .=====++++++++++++++.               
            .-=++***+++++++***++-.   :===++===.   :+++++++*-   .++++++++:  .:-+++++++++++++++++-               
            ..:--=+++++**++++*+-.    :+=.      :+++++++*-   .++++++++:     .:=++++++++++++++=.              
            .     :*++++-=++++++*-    :+=.      :++++++++-   .:::++-::.    .-: .:=++++++++++++:              
            .=====+-    :*++++..:+++++++.   :++.      ....+*:...      .=+.    ...:*+... .:=+++++++++=.             
            .+*****+:   :*++++.  =*+++++.   .+=.         .+*:         .++:    .+++++++=    .-=+++++++:             
            :+++++*+=:.:*++++.:=+++++*-    ...          .+*:         .-=.    .++++++*=      ..-=++++-             
            :+**+++**+++++++++*+++*+-.                 .++:                 .+**++**=         .:-=++.            
            .-=+*****++++++****++=.                   ....                 .::-*+-::            .::.            
            ..:-==++++++++==-:.                                             :*+                               
            :******:.                                                 .-:.                              
            .======.                                                                                    
            .::::::::::.      .::::::::::..           .:--=--:..           .-+*####=  .--:                       
            =@@@@@@@@@@@%*-.  =@@@@@@@@@@@%*-.     .=#%@@@@@@@@%*-.       -%@@@@@@@= :@@@@-   .----.             
            =@@@@#**##@@@@@#. =@@@@#***#%@@@@*.  .+@@@@@#*++*%@@@@#:      #@@@*..... .#%%#:   .%%%%:             
            =@@@@:    .+@@@@+ =@@@@-    .=@@@@+ .*@@@@+.     .:#@@@@-  -==%@@@*====. .====. -==@@@@+===:         
            =@@@@:     -@@@@* =@@@@-     -@@@@* =@@@@+  .....  .%@@@%. %@@@@@@@@@@@- :@@@@. %@@@@@@@@@@=         
            =@@@@-::::=%@@@@: =@@@@+--==*@@@@%: #@@@@: ........ +@@@@- ---%@@@*----. :@@@@. --=@@@@*---.         
            =@@@@@@@@@@@@@%-  =@@@@@@@@@@@@#+.  #@@@@. ........ =@@@@-    #@@@=      :@@@@.   .@@@@-             
            =@@@@%%%###*+-.   =@@@@*++#@@@%:    +@@@@=  ...... .#@@@@.    #@@@=      :@@@@.   .@@@@=             
            =@@@@:            =@@@@-  .*@@@%-   .#@@@@=.      .*@@@@=     #@@@=      :@@@@.   .@@@@-             
            =@@@@:            =@@@@-   .*@@@@=   .*@@@@%*===+*@@@@@=      #@@@=      :@@@@.   .%@@@#==+:         
            =@@@@:            =@@@@-     +@@@@+.   -*%@@@@@@@@@@%+:       %@@@=      :@@@@.    =%@@@@@@*         
            :===-.            :====.      -====.     .:-=++++=-:.         -===:      .===-.     .-=++=-:         
            )";                                                                                                                                                                                    


            return icon;
    };
};
