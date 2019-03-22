import pandas as pd 
import os
from src.config import data_check_path,data_prod_path,root_dir


def find_file_dirs(file_dir):
	"""
	:param file_dir:
	:return:
	"""
	dirs_list = []
	paths = os.listdir(file_dir)
	
	for path in paths:
		dirs_list.append(os.path.join(file_dir,path))

	return dirs_list


def find_csv_file(csv_path,dotname='csv'):
	"""
	:param csv_path:
	:param dotname:
	:return:
	"""
	L = []
	for root, dirs, files in os.walk(csv_path):
		for file in files:
			if file.split('.')[-1] == dotname:
				L.append(os.path.join(root, file))
	if len(L) != 4:
		return False,sorted(L)
	elif len(L) == 4:
		return True,sorted(L)


def init_path(event_length):
	"""
	:param event_length:
	:return:
	"""
	print(data_prod_path)
	print(data_check_path)

	name_app = root_dir+'result_event_length_'+str(event_length)+'/'
	print(name_app)
	if not os.path.exists(name_app):
		os.makedirs(name_app)

	return name_app


def data_prepare(name_app,event_length):
	"""
	:param name_app:
	:param event_length:
	:return:
	"""
	
	# define name parameters
	data_feature_list = ['Acceleration','Velocity','Steering_Wheel_Angle','Yaw_Rate']
	alpha_name = ['alpha','0.5']
	lambda_name = ['lambda','1.5']

	dirs_list = find_file_dirs(data_prod_path)
	data_results = pd.DataFrame()

	for dir_name in dirs_list:
		state_flag,csv_file_path = find_csv_file(dir_name)
		print(len(csv_file_path))

		if state_flag:
			csv_file_path = [x for x in csv_file_path if x.split('/')[-1].split('_')[-3]==alpha_name[-1]]
			print(csv_file_path)

			assert len(csv_file_path) == len(data_feature_list)

			count_flag = 0
			data_app = pd.DataFrame()
			for data_path in csv_file_path:
				feature_name_app = data_path.split('.')[0].split('/')[-1].split('_')[1][0:3]
				Car_ID = data_path.split('.')[0].split('/')[-1].split('_')[0]
				data = pd.read_csv(data_path).iloc[:,1:]
				data = data.replace('NOT_FULFILLMENT',0.0)
				data['divergence_score'] = data['divergence_score'].astype(float)

				data.columns = ['Time','divergence_score_' + str(feature_name_app)]
				data['divergence_score_'+str(feature_name_app)] = data['divergence_score_'+str(feature_name_app)]*(-1000)
				
				if count_flag == 0:
					data_app = data
					count_flag = 1
				else:
					data_app = pd.merge(data_app,data,on = ['Time'],how ='right')

				# print(data.head())

			data_orig = pd.read_csv(data_check_path+Car_ID+'/'+dir_name.split('/')[-1]+'.csv')
			data_result = pd.merge(data_orig,data_app,on = ['Time'],how='left')
			# standard of acd
			
			data_result['ds_total'] = data_result['divergence_score_Acc']+data_result['divergence_score_Yaw']+data_result['divergence_score_Vel'] + data_result['divergence_score_Ste']
			
			data_results = data_results.append(data_result)

			if len(data_results) >= event_length:
				top5_ds = find_top5(data_results,event_length)
				print('FINAL THRESHOLD IS:: ', top5_ds)

		else:
			continue

	data_results['is_acp'] = data_results['ds_total'].map(lambda x:1 if x>=top5_ds else 0)
	
	data_results = data_results.reset_index().drop(['index'],axis=1)
	data_results.to_csv(name_app+'drive_event_result.csv')
	print('***** SAVING ORIGINAL EVENT DATA AS '+name_app+'drive_event_result.csv'+'*****')
	save_event(data_results,name_app)


def find_top5(data_result,event_length):
	"""
	:param data_result:
	:param event_length:
	:return:
	"""
	assert len(data_result) >= event_length

	data_result = data_result.sort_values(by=['ds_total'],ascending=False)
	data_result = data_result.reset_index().drop(['index'],axis=1).fillna(0.0)

	top5 = int(0.05*len(data_result))
	print('Event Length {0} Top 5 index is {1} ,and real data length of estimating is {2} '.format(event_length,top5,len(data_result)))

	top5_ds = data_result.iloc[top5,-2]
	print('top5 divergence_score ',top5_ds)

	return top5_ds


def save_event(data,name_app):
	"""
	:param data:
	:param name_app:
	:return:
	"""
	data_re = data[data.is_acp ==1]
	events_index = [[int(x)-149, int(x)+1]for x in list(data_re.index)]
	event_df = pd.DataFrame(events_index)
	event_df.columns = ['start_index','end_index']

	event_df.to_csv(name_app+'drive_event_index.csv')
	print('*****  SAVE DRIVE EVENT INDEX AS '+name_app+'drive_event_index.csv'+' *****')


def main():
	"""
	:return:
	"""
	event_length = 10000
	
	name_app = init_path(event_length=event_length)
	data_prepare(name_app,event_length)


if __name__ == '__main__':
	main()
	






	
	
	
   