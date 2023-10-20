import React, { useState } from 'react';
import { useEffect, useRef } from 'react';
import axios from 'axios';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCog } from '@fortawesome/free-solid-svg-icons';
import './App.css';
import 'font-awesome/css/font-awesome.min.css';
import { faTimes } from '@fortawesome/free-solid-svg-icons';
import logo from './capstonelogo.png';
import { io } from 'socket.io-client';
function App() {
  useEffect(() => {
      // Connect to the server
      const socket = io('http://localhost:5000/status');
      // Register event listener for 'response' event from the server
      socket.on('response', (data) => {console.log(data); // { data: '...' }
      setSocketData(data.data);
      // Update the UI based on received data
  });
  // Clean up the effect
  return () => socket.disconnect();
  }, []);
  const [socketData, setSocketData] = useState(null);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [query, setQuery] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState({
    summarization_model: "BART",
    enable_entity_recognition: "ON",
    response_model: 'meta-llama/Llama-2-7b-chat-hf'
  });
  const [questions, setQuestions] = useState(Array(5).fill(''));
  const [responses, setResponses] = useState(Array(5).fill(''));
  const [showExamples, setShowExamples] = useState(false);
  const [examples, setExamples] = useState({ example_1: '', example_2: '', example_3: '' });

  const updateHistory = (newQuestion, newResponse) => {
    setQuestions(prev => [newQuestion, ...prev.slice(0, 4)]);
    setResponses(prev => [newResponse, ...prev.slice(0, 4)]);
  };
  const fetchExamples = async () => {
  try {
      const response = await axios.get('http://127.0.0.1:5000/api/example');
      setExamples(response.data);
      setShowExamples(true); // Show popup after receiving examples.
    } catch (error) {
      console.error("Error:", error);
    }
  };
  const handleExampleClick = async () => {
        try {
            const response = await axios.get('http://127.0.0.1:5000/api/example');
            setExamples(response.data);
            setShowExamples(true);
        } catch (error) {
            console.error("Error:", error);
        }
    };

    const exampleRef = useRef();

    const closeExamples = (e) => {
        if (e.target === exampleRef.current) setShowExamples(false);
    }

    useEffect(() => {
        window.addEventListener('click', closeExamples);

        return () => {
            window.removeEventListener('click', closeExamples);
        };
    }, []);

  const handleSubmit = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:5000/api/ask', { question,
          enable_entity_recognition: settings.enable_entity_recognition ? "on" : "off"
      });
      setAnswer(response.data.response);
      setQuery(response.data.prediction);
      updateHistory(response.data.prediction, response.data.response);
    } catch (error) {
      console.error("Error:", error);
    }
  };
  const clearHistory = () => {
    setQuestions(Array(5).fill(''));
    setResponses(Array(5).fill(''));
  };

  const handlePQItemClick = (qIndex) => {
    setQuestion(questions[qIndex]);
    setAnswer(responses[qIndex]);
  };
  const truncateTo40Words = (str) => {
    const words = str.split(' ');
    if (words.length <= 40) return str;
    return words.slice(0, 40).join(' ') + '...';
  };

  return (
    <div className="App">
        <div className="header">
            <div className="title"></div>
            <div className="settings-icon" onClick={() => setShowSettings(!showSettings)}>
                  <FontAwesomeIcon icon={faCog} />
                  <span>Settings</span>
            </div>
        </div>
        <div className="sidebar">
            <div className="past-questions-header">Past Questions</div>
            {questions.map((q, index) => (
                <div
                    key={index}
                    className={`pq-item pq${index + 1}`}
                    onClick={() => handlePQItemClick(index)} // Add this line
                >
                    <i className="fas fa-comment-dots"></i>
                    {q.split(' ').slice(0, 10).join(' ')}
                </div>
            ))}
            <div class="clear-history-btn">
                <button class="submitbutton" onClick={clearHistory}>Clear History</button>
            </div>
        </div>
        <div className="centered-content">
            <div className="image-container">
                <img src={logo} alt="Capstone Logo" />
            </div>
            {/*<div className="input-container">*/}
                {/*<div className="input-label">Describe your concerns or questions below:</div>*/}
            <div className="question-section">
                <input
                  type="text"
                  value={question}
                  onChange={e => setQuestion(e.target.value)}
                  placeholder="Ask your question here..."
                />
                <button onClick={handleSubmit}>Submit</button>

            </div>
            <button className="example_button" onClick={handleExampleClick}>Show Me Some Example Inputs</button>
            <textarea
                className="query-section"
                value={query}
                readOnly
                placeholder="Your summarized query will appear here..."
            />
            {showExamples && (
                    <div className="example-popup" ref={exampleRef}>
                        <div className="example-container">
                            <div className="close-icon" onClick={() => setShowExamples(false)}>
                                <FontAwesomeIcon icon={faTimes} />
                            </div>
                            <div className="example-block">
                                <div onClick={() => {
                                        setQuestion(examples.example_1);
                                        setShowExamples(false);
                                    }}>{truncateTo40Words(examples.example_1)}
                                </div>
                                <div onClick={() => {
                                        setQuestion(examples.example_2);
                                        setShowExamples(false);
                                    }}>{truncateTo40Words(examples.example_2)}
                                </div>
                                <div onClick={() => {
                                        setQuestion(examples.example_3);
                                        setShowExamples(false);
                                    }}>{truncateTo40Words(examples.example_3)}
                                </div>
                            </div>
                        </div>
                    </div>
            )}
            {/*</div>*/}
            <textarea
                className="answer-section"
                value={answer}
                readOnly
                placeholder="Your answer will appear here..."
            />
        </div>

        {showSettings && (
            <div className="settings">
              <div className="setting-item">
                <label>Summarization Model</label>
                <select
                  value={settings.summarization_model}
                  onChange={(e) =>
                    setSettings({ ...settings, summarization_model: e.target.value })
                  }
                >
                  <option value="bart">BART</option>
                  <option value="falcom">FALCON</option>
                  <option value="llama2">LLAMA2</option>
                </select>
              </div>

              <div className="setting-item">
                 <label>Enable Entity Recognition</label>
                 <select
                    value={settings.enable_entity_recognition ? "on" : "off"}
                    onChange={(e) =>
                        setSettings({ ...settings, enable_entity_recognition: e.target.value === "on" })
                    }
                 >
                    <option value="on">On</option>
                    <option value="off">Off</option>
                 </select>
              </div>

              <div className="setting-item">
                <label>Response Model</label>
                <select
                  value={settings.response_model}
                  onChange={(e) =>
                    setSettings({ ...settings, response_model: e.target.value })
                  }
                >
                  <option value="meta-llama/Llama-2-7b-chat-hf">Default LLAMA Model 7B</option>
                  {/*<option value="modelB">Model B</option>*/}
                </select>
              </div>
            </div>
        )}
        {socketData && (
         <div className="socket-panel">
            {socketData}
         </div>
        )}
    </div>
  );
}

export default App;
