import argparse
import models
import pickle


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='Course Project')
	parser.add_argument('-m','--model', type=int, 
		help='svm=1, decision tree=2, random forest=3', default=3)
	args = parser.parse_args()
	
	with open("../features/dpc_data.pkl", "rb") as fil:
		data = pickle.load(fil)
	
	if args.model==1:
		models.m_svm(data)
	elif args.model==2:
		models.m_dt(data)
	else:
		models.m_rf(data)