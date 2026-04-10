
const FAQAccordion = ({ question, answer }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className={aq-accordion }>
      <button className="faq-question" onClick={() => setIsOpen(!isOpen)} type="button">
        <span>{question}</span>
        <svg 
          className="faq-triangle" 
          width="16" height="16" viewBox="0 0 24 24" 
          fill="currentColor" 
          style={{ transform: isOpen ? 'rotate(180deg)' : 'rotate(90deg)', transition: 'transform 0.3s ease' }}
        >
          <path d="M7 4L19 12L7 20Z" />
        </svg>
      </button>
      <div 
        className="faq-answer-wrapper" 
        style={{ 
          maxHeight: isOpen ? '500px' : '0', 
          overflow: 'hidden', 
          transition: 'max-height 0.4s ease, opacity 0.4s ease',
          opacity: isOpen ? 1 : 0
        }}
      >
        <div className="faq-answer">
          {answer}
        </div>
      </div>
    </div>
  );
};
