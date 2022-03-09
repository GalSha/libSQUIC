if(active_index_set_temp_existing.size()>0){
				std::cout<<"EXISTING"<<std::endl;
			}
			for (size_t temp_i = 0; temp_i < active_index_set_temp_existing.size(); temp_i++)
			{
				std::cout<<active_index_set_temp_existing[temp_i].i<< "," <<  active_index_set_temp_existing[temp_i].j << "=" <<grad_temp_existing[temp_i] << std::endl;
			}

			if(active_index_set_temp_new.size()>0){
				std::cout<<"NEW"<<std::endl;
			}
			for (size_t temp_i = 0; temp_i < active_index_set_temp_new.size(); temp_i++)
			{
				std::cout<<active_index_set_temp_new[temp_i].i<< "," <<  active_index_set_temp_new[temp_i].j << "=" <<grad_temp_new[temp_i] << std::endl;
			}

			// search the top 'max_off_diag_per_row' values 
			// first sort the values by form large to small
			std::vector<integer> sort_key(grad_temp_new.size());
  			std::iota(sort_key.begin(), sort_key.end(), 0);
			std::stable_sort(sort_key.begin(), sort_key.end(),[&grad_temp_new](integer i1, integer i2) {return fabs(grad_temp_new[i1]) > fabs(grad_temp_new[i2]);});

			
			for (size_t temp_i = 0; temp_i < grad_off_diag_temp.size(); temp_i++)
			{
				std::cout<< "---> "<<   active_index_set_off_diag_temp[sort_key[temp_i]].i<< "," <<  active_index_set_off_diag_temp[sort_key[temp_i]].j << "=" <<grad_off_diag_temp[sort_key[temp_i]] << std::endl;
			}

			/*s
			
			// it may be the number off nnz in a column/row are < max_off_diag_per_row... we select the min of the two
			integer nnz_per_row = std::min( max_off_diag_per_row, integer(active_index_set_off_diag_temp.size()));

			// Add selected active indecies
			for (integer ii = 0; ii < nnz_per_row; ii++)
			{
				active_index_set.push_back(active_index_set_off_diag_temp[sort_key[ii]]);
			}

			// add the digional (always added)
			active_index.i = j;
			active_index.j = j;
			active_index_set.push_back(active_index);

			*/
