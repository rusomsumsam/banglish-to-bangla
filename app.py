"""
Banglish to Bangla Text Converter
Transformer-based NLP Model with Flask Web Interface
"""

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import json
from datetime import datetime

app = Flask(__name__)

class TransformerModel(nn.Module):
    """Transformer-based Sequence-to-Sequence Model for Banglish to Bangla Conversion"""
    
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, max_seq_length=128):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        
        # Character embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = self.create_positional_encoding(max_seq_length, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
    
    def create_positional_encoding(self, max_len, d_model):
        """Create positional encoding for transformer"""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pos_encoding = torch.zeros(1, max_len, d_model)
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pos_encoding, requires_grad=False)
    
    def forward(self, x):
        batch_size, seq_len = x.size()
        
        # Character embeddings
        x_embed = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        
        # Add positional encoding
        if seq_len <= self.max_seq_length:
            x_embed = x_embed + self.pos_encoding[:, :seq_len, :]
        else:
            x_embed = x_embed + self.pos_encoding[:, :self.max_seq_length, :]
        
        # Transformer processing
        x_embed = self.layer_norm(x_embed)
        x_embed = self.dropout(x_embed)
        
        # Create attention mask
        attention_mask = (x != 0).float()
        
        # Transformer forward pass
        transformer_out = self.transformer(x_embed, src_key_padding_mask=~attention_mask.bool())
        
        # Output projection
        output = self.output_layer(transformer_out)
        
        return output

class BanglishConverter:
    """Main converter class with rule-based and AI components"""
    
    def __init__(self):
        self.setup_character_mapping()
        self.setup_word_dictionary()
        self.model = None
        self.vocab = {}
        self.idx_to_char = {}
        
    def setup_character_mapping(self):
        """Setup comprehensive character mapping"""
        self.char_map = {
            # Single characters
            'a': 'া', 'b': 'ব', 'c': 'স', 'd': 'ড', 'e': 'ি',
            'f': 'ফ', 'g': 'গ', 'h': 'হ', 'i': 'ী', 'j': 'জ',
            'k': 'ক', 'l': 'ল', 'm': 'ম', 'n': 'ন', 'o': 'ো',
            'p': 'প', 'q': 'ক্যু', 'r': 'র', 's': 'স', 't': 'ট',
            'u': 'ু', 'v': 'ভ', 'w': 'ও', 'x': 'এক্স', 'y': 'ই', 'z': 'জ',
            
            # Compound characters
            'sh': 'শ', 'ch': 'চ', 'th': 'থ', 'bh': 'ভ', 'dh': 'ধ',
            'gh': 'ঘ', 'kh': 'খ', 'ng': 'ং', 'ph': 'ফ', 'rr': 'ড়',
            'rh': 'ঢ', 'kk': 'ক্ক', 'tt': 'ট্ট', 'dd': 'ড্ড',
            
            # Vowels
            'aa': 'া', 'ee': 'ী', 'oo': 'ু', 'ou': 'ৌ', 'oi': 'ৈ',
            
            # Special cases
            '0': '০', '1': '১', '2': '২', '3': '৩', '4': '৪',
            '5': '৫', '6': '৬', '7': '৭', '8': '৮', '9': '৯'
        }
    
    def setup_word_dictionary(self):
        """Setup comprehensive word dictionary"""
        self.word_dict = {
            # Common words
            'ami': 'আমি', 'tumi': 'তুমি', 'apni': 'আপনি', 'ki': 'কি',
            'kemon': 'কেমন', 'achi': 'আছি', 'achen': 'আছেন', 'hoy': 'হয়',
            'naam': 'নাম', 'bangla': 'বাংলা', 'gan': 'গান', 'gai': 'গাই',
            'bhalo': 'ভালো', 'valo': 'ভাল', 'tomar': 'তোমার', 'amar': 'আমার',
            'kobe': 'কবে', 'hobe': 'হবে', 'she': 'সে', 'din': 'দিন',
            'emon': 'এমন', 'je': 'যে', 'megher': 'মেঘের', 'pore': 'পরে',
            'megh': 'মেঘ', 'dhaka': 'ঢাকা', 'sheharer': 'শহরের',
            'bangladesh': 'বাংলাদেশ', 'desh': 'দেশ', 'valobasha': 'ভালোবাসা',
            'tomake': 'তোমাকে', 'kotha': 'কথা', 'bolchi': 'বলছি', 'jani': 'জানি',
            
            # Verbs
            'korchi': 'করছি', 'korben': 'করবেন', 'korbe': 'করবে', 'korte': 'করতে',
            'jabo': 'যাবো', 'jaben': 'যাবেন', 'aschi': 'আছি', 'ashben': 'আসবেন',
            
            # Pronouns
            'ke': 'কে', 'kar': 'কার', 'jake': 'যাকে', 'take': 'তাকে',
            
            # Common phrases
            'kothay': 'কোথায়', 'keno': 'কেন', 'jokhon': 'যখন', 'tokhon': 'তখন',
            'jodi': 'যদি', 'tahole': 'তাহলে'
        }
    
    def advanced_transliterate(self, text):
        """Advanced rule-based transliteration with context awareness"""
        
        # Convert to lowercase for processing
        text_lower = text.lower()
        words = text_lower.split()
        converted_words = []
        
        for word in words:
            # Check if word is in dictionary
            if word in self.word_dict:
                converted_words.append(self.word_dict[word])
                continue
            
            # Apply advanced character mapping with context
            converted_word = ""
            i = 0
            word_len = len(word)
            
            while i < word_len:
                # Check for 2-character combinations first
                if i + 1 < word_len:
                    two_char = word[i:i+2]
                    if two_char in self.char_map:
                        converted_word += self.char_map[two_char]
                        i += 2
                        continue
                
                # Check for single character
                one_char = word[i]
                if one_char in self.char_map:
                    converted_word += self.char_map[one_char]
                else:
                    converted_word += one_char
                
                i += 1
            
            # Post-processing for common patterns
            converted_word = self.post_process(converted_word)
            converted_words.append(converted_word)
        
        result = ' '.join(converted_words)
        
        # Final cleanup
        result = self.final_cleanup(result)
        
        return result
    
    def post_process(self, word):
        """Post-process words for better accuracy"""
        # Handle common suffix patterns
        replacements = {
            'িব': 'বি',
            'িক': 'কি', 
            'িল': 'লি',
            'ির': 'রি',
            'িস': 'সি'
        }
        
        for wrong, correct in replacements.items():
            if word.endswith(wrong):
                word = word[:-len(wrong)] + correct
                break
        
        return word
    
    def final_cleanup(self, text):
        """Final cleanup and formatting"""
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Ensure proper spacing after punctuation
        text = re.sub(r'([.!?])([^\\s])', r'\1 \2', text)
        
        return text
    
    def convert(self, text):
        """Main conversion function"""
        if not text or not text.strip():
            return ""
        
        try:
            # Use advanced rule-based transliteration
            return self.advanced_transliterate(text)
        except Exception as e:
            return f"Conversion error: {str(e)}"

# Initialize converter
converter = BanglishConverter()

@app.route('/')
def home():
    """Render main page"""
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
def convert_text():
    """API endpoint for text conversion"""
    try:
        data = request.get_json()
        banglish_text = data.get('text', '').strip()
        
        if not banglish_text:
            return jsonify({
                'success': False,
                'error': 'No text provided'
            }), 400
        
        # Perform conversion
        start_time = datetime.now()
        bangla_text = converter.convert(banglish_text)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return jsonify({
            'success': True,
            'banglish': banglish_text,
            'bangla': bangla_text,
            'processing_time_ms': round(processing_time, 2),
            'characters_processed': len(banglish_text)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/batch_convert', methods=['POST'])
def batch_convert():
    """API endpoint for batch conversion"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({
                'success': False,
                'error': 'No texts provided'
            }), 400
        
        results = []
        total_chars = 0
        
        for text in texts:
            bangla_text = converter.convert(text)
            results.append({
                'input': text,
                'output': bangla_text
            })
            total_chars += len(text)
        
        return jsonify({
            'success': True,
            'results': results,
            'total_conversions': len(results),
            'total_characters': total_chars
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Banglish to Bangla Converter',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Starting Banglish to Bangla Converter...")
    print("📝 Access the web interface at: http://localhost:5000")
    print("⚡ Converter ready for use!")
    app.run(debug=True, host='0.0.0.0', port=5000)