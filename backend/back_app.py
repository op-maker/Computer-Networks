from model import load_model_and_tokenizer, predict_relationship, predict_zero_shot
from flask import Flask, jsonify, make_response, request

model, tokenizer = load_model_and_tokenizer()

app = Flask(__name__)

@app.route('/api/v1.0/pred_logical_rel', methods=['POST'])
def pred_relationship():
    posted_data = request.get_json()
    text1, text2 = posted_data['text1'], posted_data['text2']

    if len(text1) == 0 or len(text2) == 0:
        return make_response(jsonify({'code': 'INCORRECT INPUT'}), 400)

    result = predict_relationship(text1, text2, model, tokenizer)
    return make_response(jsonify(result), 200)

@app.route('/api/v1.0/pred_zero_shot', methods=['POST'])
def pred_zero_shot():
    posted_data = request.get_json()
    classes, text = posted_data['classes'], posted_data['text']
    all_classes = [cl for cl in classes.split('\n') if len(cl) != 0]
    
    if len(all_classes) == 0 or len(text) == 0:
        return make_response(jsonify({'code': 'INCORRECT INPUT'}), 400)
    
    result = predict_zero_shot(text, all_classes, model, tokenizer)
    
    return make_response(jsonify(result), 200)



@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'code': 'PAGE_NOT_FOUND'}), 404)

@app.errorhandler(500)
def server_error(error):
    return make_response(jsonify({'code': 'INTERNAL_SERVER_ERROR'}), 500)


if __name__ == '__main__':
    app.run(debug=True)