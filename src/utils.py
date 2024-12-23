import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    try:
        # Extract the directory path from the file path
        dir_path = os.path.dirname(file_path)
        
        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        
        # Save the object to the specified file path
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e: 
        raise CustomException(e, sys)




# -------> Another save_object function
# def save_object(file_path, obj):
#     try:
#         dir_path = os.path.dirname(file_path)
#         os.makedirs(dir_path, exist_ok=True)

#         # Explicitly check file access
#         with open(file_path, 'wb') as file_obj:
#             pickle.dump(obj, file_obj)
#             logging.info(f"Object successfully saved at {file_path}")
#     except PermissionError as pe:
#         logging.error(f"Permission denied: {pe}")
#         raise CustomException(pe, sys)
#     except Exception as e:
#         raise CustomException(e, sys)



def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e, sys)
