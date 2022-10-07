#include <cstdio>
#include <iostream>




void make_files(){
    for(int i=0;i<4;i++){// 4 files
        ofstream myfile;
        char buffer[32]; // The filename buffer.
        snprintf(buffer, sizeof(char) * 32, "data_txt/data%02d.txt", i);
        myfile.open(buffer);
        myfile << "egamma\tcostheta\tenergy_g1\tenergy_g2\ttheta_g1\ttheta_g2\tphi_g1\tphi_g2\ttheta_p\tphi_p\tclustersize_g1\tclustersize_g2\tclusterPEDcount_g1\tclusterPEDcount_g2\tclustersize_p\tclusterPEDcount_p\n";
        myfile.close();
    }
}


void fill_files(TTree* t0,TTree* t1,TTree* t2,TTree* t3){
//loop over all trees
TTree* t[4]={t0,t1,t2,t3};

    for(int i=0;i<4;i++){
        
        //set branch adresses of tree vars    
        double egamma=0.;
        t[i]->SetBranchAddress("egamma",&egamma);
        double costheta=0.;
        t[i]->SetBranchAddress("costheta",&costheta);
        double energy_g1=0.;
        t[i]->SetBranchAddress("energy_g1",&energy_g1);
        double energy_g2=0.;
        t[i]->SetBranchAddress("energy_g2",&energy_g2);
        double theta_g1=0.;
        t[i]->SetBranchAddress("theta_g1",&theta_g1);
        double theta_g2=0.;
        t[i]->SetBranchAddress("theta_g2",&theta_g2);
        double phi_g1=0.;
        t[i]->SetBranchAddress("phi_g1",&phi_g1);
        double phi_g2=0.;
        t[i]->SetBranchAddress("phi_g2",&phi_g2);
        double theta_p=0.;
        t[i]->SetBranchAddress("theta_p",&theta_p);
        double phi_p=0.;
        t[i]->SetBranchAddress("phi_p",&phi_p);
        double clustersize_g1=0.;
        t[i]->SetBranchAddress("clustersize_g1",&clustersize_g1);
        double clustersize_g2=0.;
        t[i]->SetBranchAddress("clustersize_g2",&clustersize_g2);
        double clusterPEDcount_g1=0.;
        t[i]->SetBranchAddress("clusterPEDcount_g1",&clusterPEDcount_g1);
        double clusterPEDcount_g2=0.;
        t[i]->SetBranchAddress("clusterPEDcount_g2",&clusterPEDcount_g2);
        double clustersize_p=0.;
        t[i]->SetBranchAddress("clustersize_p",&clustersize_p);
        double clusterPEDcount_p=0.;
        t[i]->SetBranchAddress("clusterPEDcount_p",&clusterPEDcount_p);
        
        
        char buffer[32]; // The filename buffer.
        snprintf(buffer, sizeof(char) * 32, "data_txt/data%02d.txt", i);
        ofstream myfile;
        myfile.open(buffer,std::ios::app);
        
        for(int j=0;j<t[i]->GetEntries();j++){//tree entries
            t[i]->GetEntry(j);
            myfile << egamma <<"\t"<< costheta <<"\t"<< energy_g1 <<"\t"<< energy_g2 <<"\t"<< theta_g1 <<"\t"<< theta_g2 <<"\t"<< phi_g1 <<"\t"<< phi_g2 <<"\t"<< theta_p <<"\t"<< phi_p <<"\t"<< clustersize_g1 <<"\t"<< clustersize_g2 <<"\t"<< clusterPEDcount_g1 <<"\t"<< clusterPEDcount_g2 <<"\t"<< clustersize_p <<"\t"<< clusterPEDcount_p << "\n";              
        }
        myfile.close();

    }
}


void create_csv(){
    TFile* f0 = new TFile("root/pi0.root","READ");
    TTree* t0 = (TTree *)f0->Get("myMLtree_cut");
    
    TFile* f1 = new TFile("root/2pi0.root","READ");
    TTree* t1 = (TTree *)f1->Get("myMLtree_cut");

    TFile* f2 = new TFile("root/eta.root","READ");
    TTree* t2 = (TTree *)f2->Get("myMLtree_cut");
    
    TFile* f3 = new TFile("root/etap.root","READ");
    TTree* t3 = (TTree *)f3->Get("myMLtree_cut");
    
    int count=0;
    make_files();
    fill_files(t0,t1,t2,t3);
}

