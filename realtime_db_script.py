import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

databaseURL = 'https://stonks-7b3ab-default-rtdb.firebaseio.com/'
# Fetch the service account key JSON file 
cred = credentials.Certificate('stonks-7b3ab-firebase-adminsdk-h2p7m-7106810010.json')
# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred, {
    'databaseURL': databaseURL
})



def add_new_data(data):
    ''' Function to add a new row of prediction to specific 
    document_id of specific stock_symbol
    '''
    ref = db.reference('/')
    ref.set(data)

def update(stock_symbol, stock_pred_id, field, updated_value):
    ''' Function to update specific value of specific field of specific document_id of specific stock_symbol
    '''
    ref = db.reference('stocks')
    stock_ref = ref.child(stock_symbol).child(stock_pred_id)
    stock_ref.update({
    field : updated_value
})

def retreive(stock_symbol):
    ''' Function to retreive all the data of a specific stock_symbol
    '''
    ref = db.reference('stocks')
    return ref.child(stock_symbol).get()

def main():
    stock_symbol_1 = 'cipla' # placeholder
    prediction_date_1 = '17-03-2021' # placeholder value for the date which's Close price is predicted by the
    # ML model
    predicted_close_1 = 20 # placeholder value to; the number that the ML model churns out
    stock_pred_id_1 = 'cipla_18032021' # (stock_symbol + _ + ddmmyy)
    actual_open_1 = 12 # placeholder
    stock_info_1 = 'cipla_stock_info' # placeholder (stock_symbol + _ + 'stock_info')
    data_1 = {
        'stocks': 
        {
            stock_symbol_1:
            {
                stock_pred_id_1:
                {
                    'open': actual_open_1, 
                    'close': predicted_close_1,
                    'prediction_for': prediction_date_1  # placeholder values for now
                }
            }
        }
    }
    add_new_data(data_1)

    print(retreive(stock_symbol_1))

if __name__ == '__main__':
    main()